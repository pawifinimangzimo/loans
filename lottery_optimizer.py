#!/usr/bin/env python3
"""
COMPLETE LOTTERY OPTIMIZER WITH DASHBOARD
- SQLite storage
- Pandas analysis
- Self-contained HTML dashboard
"""

import sqlite3
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Union
import logging

# ======================
# DEFAULT CONFIGURATION
# ======================
DEFAULT_CONFIG = {
    'data': {
        'db_path': 'data/lottery.db',
        'historical_csv': 'data/historical.csv',
        'results_dir': 'results/'
    },
    'analysis': {
        'hot_days': 30,
        'cold_threshold': 60,
        'top_range': 10,  # Now using top_range instead of top_n_results
        'min_display_matches': 1,
        'combination_analysis': {
            'pairs': True,
            'triplets': True,
            'quadruplets': False,
            'quintuplets': False,
            'sixtuplets': False
        }
    },
    'strategy': {
        'numbers_to_select': 6,
        'number_pool': 55
    }
}

# ======================
# CORE ANALYZER CLASS
# ======================
class LotteryAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self._validate_config() 
        # Add validation
        if 'sets_to_generate' not in self.config['output']:
            self.config['output']['sets_to_generate'] = 4
        # Rest of your init code
        #number pool initialization
        self.number_pool = list(range(1, config['strategy']['number_pool'] + 1))
        self.weights = pd.Series(1.0, index=self.number_pool) 
        #number pool initialization end 
        #mode handler 
        
        self._validate_overdue_analysis_config()  # Add this line
        self.conn = self._init_db()
        self._init_mode_handler()  # Add this line

        self._prepare_filesystem()

    # ======================
    # NEW CONFIG VALIDATION
    # ======================
    def _validate_config(self):
        """Validate combination analysis config"""
        if 'combination_analysis' not in self.config['analysis']:
            return
            
        valid_sizes = {'pairs', 'triplets', 'quadruplets', 'quintuplets', 'sixtuplets'}
        invalid = [
            size for size in self.config['analysis']['combination_analysis']
            if size not in valid_sizes
        ]
        
        if invalid:
            raise ValueError(
                f"Invalid combination_analysis keys: {invalid}. "
                f"Valid options are: {valid_sizes}"
            )

    def _init_db(self) -> sqlite3.Connection:
        """Initialize SQLite database with optimized schema"""
        Path(self.config['data']['db_path']).parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(self.config['data']['db_path'])
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS draws (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                n1 INTEGER, n2 INTEGER, n3 INTEGER,
                n4 INTEGER, n5 INTEGER, n6 INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON draws(date)")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_numbers 
            ON draws(n1,n2,n3,n4,n5,n6)
        """)
        
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_primes ON draws(n1,n2,n3,n4,n5,n6)
            WHERE n1 IN (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53);
            
            CREATE INDEX IF NOT EXISTS idx_low_numbers ON draws(n1,n2,n3,n4,n5,n6)
            WHERE n1 <= 10 OR n2 <= 10 OR n3 <= 10 OR n4 <= 10 OR n5 <= 10 OR n6 <= 10;
        """)

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS number_overdues (
                number INTEGER PRIMARY KEY,
                last_seen_date TEXT,
                current_overdue INTEGER DEFAULT 0,
                avg_overdue REAL,
                max_overdue INTEGER,
                is_overdue BOOLEAN DEFAULT FALSE
            );
            CREATE INDEX IF NOT EXISTS idx_overdue ON number_overdues(is_overdue);
        """)        
        
        return conn

    def _prepare_filesystem(self) -> None:
        """Ensure required directories exist"""
        Path(self.config['data']['results_dir']).mkdir(exist_ok=True)

    def load_data(self) -> None:
        """Load CSV data into SQLite with validation"""
        try:
            df = pd.read_csv(
                self.config['data']['historical_csv'],
                header=None,
                names=['date', 'numbers'],
                parse_dates=['date']
            )
            
            # Split numbers into columns
            nums = df['numbers'].str.split('-', expand=True)
            for i in range(self.config['strategy']['numbers_to_select']):
                df[f'n{i+1}'] = nums[i].astype(int)
            
            # Validate number ranges
            pool_size = self.config['strategy']['number_pool']
            for i in range(1, self.config['strategy']['numbers_to_select'] + 1):
                invalid = df[(df[f'n{i}'] < 1) | (df[f'n{i}'] > pool_size)]
                if not invalid.empty:
                    raise ValueError(f"Invalid numbers in n{i} (range 1-{pool_size})")
            
            # Store in SQLite
            df[['date'] + [f'n{i}' for i in range(1, 7)]].to_sql(
                'draws', self.conn, if_exists='replace', index=False
            )

            # Move gap analysis initialization AFTER data is loaded
            if self.config['analysis']['overdue_analysis']['enabled']:
                self._initialize_overdue_analysis()
   
        except Exception as e:
            raise ValueError(f"Data loading failed: {str(e)}")
####################
######### Number Trends ###############

    def get_number_trend(self, num: int, lookback=10) -> dict:
        """Structured number trend analysis"""
        query = """
        WITH appearances AS (
            SELECT date, ROW_NUMBER() OVER (ORDER BY date DESC) as rn 
            FROM draws
            WHERE ? IN (n1,n2,n3,n4,n5,n6)
            ORDER BY date DESC
            LIMIT ?
        )
        SELECT date, rn FROM appearances ORDER BY date
        """
        results = self.conn.execute(query, (num, lookback)).fetchall()
        
        appearance_flags = [0] * lookback
        for date, rn in results:
            appearance_flags[rn-1] = 1  # Convert to 0-based index
            
        gaps = [results[i][1] - results[i+1][1] - 1 
                for i in range(len(results)-1)] if len(results) > 1 else [lookback]
        
        return {
            "analysis": "number_trend",
            "number": num,
            "last_n_appearances": appearance_flags,
            "current_gap": results[0][1]-1 if results else lookback,
            "average_gap": sum(gaps)/len(gaps) if gaps else float(lookback),
            "last_seen": results[0][0] if results else None,
            "appearance_count": len(results)
        }

#######################################
    def get_weights(self) -> dict:  
        """Export ALL weights, including new ones."""  
        return {  
            'frequency': self._get_frequency_weights().to_dict(),  
            'recency': self._get_recent_weights().to_dict(),  
            'gaps': self._get_gap_weights(),  
            'overdue': self._get_overdue_weights(),  
            'primes': self._get_prime_weights(),          # New  
            'odd_even': self._get_odd_even_weights(),     # New  
            'sum': self._get_sum_weights(),               # New  
            
            'time_weights': self.get_time_weights(),
            'cooccurrence': self.get_cooccurrence_weights(),
            
            'final': self.weights.to_dict()  # Blended result  
        }  

####################
    def get_frequencies(self, count: int = None) -> pd.Series:
        """Get number frequencies using optimized SQL query"""
        top_n = count or self.config['analysis']['top_range']
        query = """
            WITH nums AS (
                SELECT n1 AS num FROM draws UNION ALL
                SELECT n2 FROM draws UNION ALL
                SELECT n3 FROM draws UNION ALL
                SELECT n4 FROM draws UNION ALL
                SELECT n5 FROM draws UNION ALL
                SELECT n6 FROM draws
            )
            SELECT num, COUNT(*) as frequency 
            FROM nums 
            GROUP BY num 
            ORDER BY frequency DESC
            LIMIT ?
        """
        result = pd.read_sql(query, self.conn, params=(top_n,))
        
        # Return empty Series with correct structure if no results
        if result.empty:
            return pd.Series(dtype=float, name='frequency')
            
        return result.set_index('num')['frequency']
##############################

##################### ANALYZE GAP ###################

    def analyze_gaps(self) -> dict:
        """
        Calculate adjacent gaps and identify overdue gaps.
        Returns: {
            'frequency': {gap: count}, 
            'overdue': [{'gap': int, 'draws_overdue': int, 'avg_frequency': float}],
            'clusters': {'small_gap_clusters': int}
        }
        """
        if not self.config['analysis']['gap_analysis']['enabled']:
            return {}

        # Load config
        cfg = self.config['analysis']['gap_analysis']
        lookback = cfg['lookback_draws']
        min_gap, max_gap = cfg['min_gap_size'], cfg['max_gap_size']
        min_freq = cfg['min_frequency']

        # Load draws
        query = f"""
        SELECT n1, n2, n3, n4, n5, n6 
        FROM draws 
        ORDER BY date DESC 
        LIMIT {lookback}
        """
        draws = pd.read_sql(query, self.conn)
        sorted_draws = [sorted(row) for _, row in draws.iterrows()]

        # Calculate gaps
        gap_counts = defaultdict(int)
        gap_last_seen = {}
        small_gap_clusters = 0  # If track_consecutive=True

        for i, draw in enumerate(sorted_draws):
            gaps = [draw[j+1] - draw[j] for j in range(len(draw)-1)]
            
            # Track frequencies and last seen
            for gap in gaps:
                if min_gap <= gap <= max_gap:
                    gap_counts[gap] += 1
                    gap_last_seen[gap] = i  # Store most recent occurrence

            # Track clusters of small gaps (optional)
            if cfg['track_consecutive']:
                if sum(1 for g in gaps if g <= 5) >= 2:  # Example threshold
                    small_gap_clusters += 1

        # Filter by min_frequency
        gap_counts = {g: c for g, c in gap_counts.items() if c >= min_freq}

        # Identify overdue gaps
        overdue = []
        for gap, count in gap_counts.items():
            avg_frequency = round(lookback / count, 1)
            draws_since_seen = len(sorted_draws) - gap_last_seen[gap] - 1
            
            if cfg['mode'] == 'auto':
                is_overdue = draws_since_seen > (avg_frequency * cfg['auto_threshold'])
            else:
                is_overdue = draws_since_seen > cfg['manual_threshold']
            
            if is_overdue:
                overdue.append({
                    'gap': gap,
                    'draws_overdue': draws_since_seen,
                    'avg_frequency': avg_frequency
                })

        return {
            'frequency': dict(sorted(gap_counts.items(), key=lambda x: -x[1])),
            'overdue': overdue,
            'clusters': {'small_gap_clusters': small_gap_clusters} if cfg['track_consecutive'] else {}
        }

#####################################################
# ======================
# COMBINATION ANALYSIS 
# ======================

    def get_combinations(self, size: int = 2, verbose: bool = None ) -> pd.DataFrame:
        """Get frequency of number combinations with proper SQL ordering.
        Args:
            size: 2 for pairs, 3 for triplets, etc. (default=2)
            verbose: Whether to print status messages (default=True)
        Returns:
            DataFrame with columns [nX, nY, ..., frequency]
        """
        # ====== CONFIG VALIDATION ======
        if verbose is None:
            verbose = self.config['output'].get('verbose', True)
        
        combo_type = {2: 'pairs', 3: 'triplets', 4: 'quadruplets', 
                      5: 'quintuplets', 6: 'sixtuplets'}.get(size)
        if not combo_type:
            if verbose:
                print(f"⚠️  Invalid combination size: {size} (must be 2-6)")
            return pd.DataFrame()
        
        if not hasattr(self, 'config'):
            if verbose:
                print("⚠️  Config not loaded - combination analysis unavailable")
            return pd.DataFrame()
        
        if not self.config['analysis']['combination_analysis'].get(combo_type, False):
            if verbose:
                print(f"ℹ️  {combo_type.capitalize()} analysis disabled in config")
            return pd.DataFrame()

        # ====== PARAMETERS ======
        top_n = self.config['analysis']['top_range']
        min_count = self.config['analysis'].get('min_combination_count', 2)  # Default to 2 if missing
        cols = [f'n{i}' for i in range(1, self.config['strategy']['numbers_to_select'] + 1)]
        
        if verbose:
            print(f"🔍 Analyzing {combo_type} (min {min_count} appearances)...", end=' ', flush=True)
        # ====== QUERY GENERATION ======
        queries = []
        for combo in combinations(cols, size):
            select_cols = ', '.join(combo)
            queries.append(f"""
                SELECT {select_cols}, COUNT(*) as frequency
                FROM draws
                GROUP BY {select_cols}
                HAVING frequency >= {min_count}  
            """)
        
        full_query = " UNION ALL ".join(queries)
        full_query += f"\nORDER BY frequency DESC\nLIMIT {top_n}"

        try:
            result = pd.read_sql(full_query, self.conn)
            if verbose:
                print(f"found {len(result)} combinations")
            return result
        except sqlite3.Error as e:
            if verbose:
                print("failed")
            raise RuntimeError(f"SQL query failed: {str(e)}")

#=======================

    def get_temperature_stats(self) -> Dict:
        """Enhanced temperature stats with structured output"""
        # Existing hot/cold calculation
        hot_limit = self.config['analysis']['recency_bins']['hot']
        cold_limit = self.config['analysis']['recency_bins']['cold']
        
        hot = pd.read_sql(hot_query, self.conn)['num'].unique().tolist()
        cold = pd.read_sql(cold_query, self.conn)['num'].unique().tolist()
        
        # Additional prime classification
        primes = set(self._get_prime_numbers())
        hot_primes = [n for n in hot if n in primes]
        cold_primes = [n for n in cold if n in primes]
        
        # Apply top_range limit
        top_n = self.config['analysis']['top_range']
        hot = hot[:top_n]
        cold = cold[:top_n]
        hot_primes = hot_primes[:top_n]
        cold_primes = cold_primes[:top_n]
        
        return {
            # Structured format
            "analysis": "temperature_stats",
            "metadata": {
                "hot_threshold": hot_limit,
                "cold_threshold": cold_limit,
                "top_range": top_n
            },
            "numbers": {
                "hot": [int(n) for n in hot],
                "cold": [int(n) for n in cold]
            },
            "primes": {
                "hot_primes": [int(n) for n in hot_primes],
                "cold_primes": [int(n) for n in cold_primes]
            },
            # Backward compatible format
            "legacy_format": {
                'hot': hot,
                'cold': cold
            }
        }

    def _get_draw_count(self) -> int:
        """Get total number of draws in database."""
        return self.conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]

    def _get_analysis_draw_limit(self, feature: str, default: int) -> int:
        """NEW: Safe config reader for analysis draw counts"""
        try:
            limit = self.config['analysis'][feature].get('draws', default)
            return max(1, min(limit, self._get_draw_count()))  # Clamp to valid range
        except (KeyError, TypeError):
            return default

    def _get_historical_ratio(self) -> float:
        """Get long-term high/low ratio average"""
        query = """
        WITH all_numbers AS (
            SELECT n1 as num FROM draws UNION ALL
            SELECT n2 FROM draws UNION ALL
            SELECT n3 FROM draws UNION ALL
            SELECT n4 FROM draws UNION ALL
            SELECT n5 FROM draws UNION ALL
            SELECT n6 FROM draws
        )
        SELECT 
            SUM(CASE WHEN num > ? THEN 1 ELSE 0 END) * 1.0 /
            NULLIF(SUM(CASE WHEN num <= ? THEN 1 ELSE 0 END), 0)
        FROM all_numbers
        """
        low_max = self.config['analysis']['high_low']['low_number_max']
        return self.conn.execute(query, (low_max, low_max)).fetchone()[0]
# Helpers         

    def _get_time_weights(self, window: int) -> dict:
        """Get time weights for a specific lookback window."""
        recent_draws = self._get_draws_in_window(window)
        counts = recent_draws[[f'n{i}' for i in range(1, 7)]].stack().value_counts()
        return (counts / counts.sum()).to_dict()
        
    def _verify_overdue_analysis(self):
        """Verify overdue analysis data exists"""
        # Check if any gaps recorded at all
        total_numbers = self.conn.execute(
            "SELECT COUNT(*) FROM number_overdues"
        ).fetchone()[0]
        
        # Check max gaps recorded
        max_overdue = self.conn.execute(
            "SELECT MAX(current_overdue) FROM number_overdues"
        ).fetchone()[0]
        
        if self.config['output'].get('verbose', True):
            print(f"\nOVERDUE ANALYSIS VERIFICATION:")
            print(f"Total numbers tracked: {total_numbers}/{self.config['strategy']['number_pool']}")
            print(f"Max current gap: {max_overdue}")
            print(f"Thresholds: Auto={self.config['analysis']['overdue_analysis']['auto_threshold']}x avg, Manual={self.config['analysis']['overdue_analysis']['manual_threshold']} draws")


    def debug_overdue_status(self):
        """Temporary method to debug gap analysis"""
        query = """
        SELECT number, current_overdue, avg_overdue, is_overdue 
        FROM number_overdues 
        WHERE is_overdue = TRUE OR current_overdue > avg_overdue
        ORDER BY current_overdue DESC
        """
        df = pd.read_sql(query, self.conn)
        print("\nGAP ANALYSIS DEBUG:")
        print(df.to_string())

    def _validate_overdue_analysis_config(self):
        """Ensure gap_analysis config has all required fields"""
        overdue_config = self.config.setdefault('analysis', {}).setdefault('overdue_analysis', {})
        overdue_config.setdefault('enabled', True)  # Default to True since you're using it
        overdue_config.setdefault('mode', 'auto')
        overdue_config.setdefault('auto_threshold', 1.5)
        overdue_config.setdefault('manual_threshold', 10)
        overdue_config.setdefault('weight_influence', 0.3)

    def _get_overdue_numbers(self) -> List[int]:
        """Return list of numbers marked as overdue in number_gaps table"""
        if not self.config['analysis']['overdue_analysis']['enabled']:
            return []
        
        query = "SELECT number FROM number_overdues WHERE is_overdue = TRUE"
        return [row[0] for row in self.conn.execute(query)]
        
    def _calculate_avg_overdue(self, num):
        """Calculate average gap for a specific number"""
        gaps = self.conn.execute("""
            SELECT julianday(d1.date) - julianday(d2.date) as gap
            FROM draws d1
            JOIN draws d2 ON d1.date > d2.date
            WHERE ? IN (d1.n1, d1.n2, d1.n3, d1.n4, d1.n5, d1.n6)
              AND ? IN (d2.n1, d2.n2, d2.n3, d2.n4, d2.n5, d2.n6)
            ORDER BY d1.date DESC
            LIMIT 10
        """, (num, num)).fetchall()
        
        if not gaps:
            return 0
        return sum(gap[0] for gap in gaps) / len(gaps)
        
#======================
# Start Set generator
#======================

    def _get_sum_percentile(self, sum_value: int) -> float:
        """Calculate what percentile a sum falls into historically"""
        query = """
        WITH sums AS (
            SELECT (n1+n2+n3+n4+n5+n6) as total 
            FROM draws
        )
        SELECT 
            CAST(SUM(CASE WHEN total <= ? THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)
        FROM sums
        """
        percentile = self.conn.execute(query, (sum_value,)).fetchone()[0]
        return round(percentile * 100, 1)

    def _is_valid(self, numbers: List[int]) -> Tuple[bool, List[str]]:
        """
        Returns: 
            (is_valid, notes)
        Notes format:
            ["Sum: 180 (Optimal range: 128-195)", "2 hot numbers", ...]
        """
        notes = []
        total = sum(numbers)
        
        # 1. Sum validation
        sum_stats = self.get_sum_stats()
        if not sum_stats.get('error'):
            q1, q3 = sum_stats['q1'], sum_stats['q3']
            margin = (q3 - q1) * self.config['validation'].get('sum_margin', 0.15)
            
            if total < (q1 - margin):
                return False, [f"Sum: {total} (Below minimum {int(q1 - margin)})"]
            if total > (q3 + margin):
                return False, [f"Sum: {total} (Above maximum {int(q3 + margin)})"]

            # Insert percentile calculation here (new code)
            if self.config['output'].get('show_percentiles', True):
                percentile = self._get_sum_percentile(total)
                notes.append(
                    f"Sum: {total} (Top {percentile}% | Range: {int(q1)}-{int(q3)})"
                )
            else:
                notes.append(f"Sum: {total} (Optimal range: {int(q1)}-{int(q3)})")

        if self.config['analysis']['patterns']['odd_even']['enabled']:
            even_count = sum(1 for n in numbers if n % 2 == 0)
            if even_count == 0 or even_count == len(numbers):
                notes.append(f"Warning: All {'odd' if even_count==0 else 'even'} numbers")


        # 2. Hot numbers (optional)
        if self.config['validation'].get('check_hot_numbers', True):
            hot_nums = [n for n in numbers if n in self.get_temperature_stats()['hot']]
            if hot_nums:
                notes.append(f"{len(hot_nums)} hot numbers ({', '.join(map(str, hot_nums))})")
        
        return True, notes

    def generate_valid_sets(self) -> List[Dict]:
        """
        Returns: 
            [{
                'numbers': [7,9,...],
                'sum': 180,
                'notes': ["Sum: 180...", ...]
            }, ...]
        """
        results = []
        attempts = 0
        max_attempts = self.config['output'].get('max_attempts', 100)
        
        while len(results) < self.config['output']['sets_to_generate']:
            candidate = self._generate_candidate()
            is_valid, notes = self._is_valid(candidate)
            
            if is_valid:
                results.append({
                    'numbers': candidate,
                    'sum': sum(candidate),
                    'notes': notes
                })
            attempts += 1
            
            if attempts >= max_attempts:
                logging.warning(f"Max attempts reached ({max_attempts})")
                break
        
        return results

    def generate_sets(self, strategy: str = None) -> List[List[int]]:
        """Generate sets with sum range validation."""
        strategy = strategy or self.config.get('strategy', {}).get('default_strategy', 'balanced')
        num_sets = self.config['output'].get('sets_to_generate', 4)
        
        # Get historical sum stats
        sum_stats = self.get_sum_stats()
        if sum_stats.get('error'):
            q1, q3 = 0, 200  # Fallback ranges
        else:
            q1, q3 = sum_stats['q1'], sum_stats['q3']
        
        sets = []
        attempts = 0
        max_attempts = num_sets * 3  # Prevent infinite loops
        
        while len(sets) < num_sets and attempts < max_attempts:
            attempts += 1
            if self.mode == 'auto':
                self._init_weights()
            
            # Generate candidate set
            candidate = self._generate_candidate_set(strategy)
            total = sum(candidate)
            
            # Validate sum is within interquartile range (Q1-Q3)
            if q1 <= total <= q3:
                sets.append(sorted(candidate))
        
        return sets if sets else [self._generate_fallback_set()]

    def _generate_candidate_set(self, strategy: str) -> List[int]:
        """Generate one candidate set based on strategy."""
        if strategy == 'balanced':
            hot = self.get_temperature_stats()['hot'][:3]
            cold = self.get_temperature_stats()['cold'][:2]
            remaining = self.config['strategy']['numbers_to_select'] - len(hot) - len(cold)
            random_nums = np.random.choice(
                [n for n in self.number_pool if n not in hot + cold],
                size=remaining,
                replace=False
            )
            return hot + cold + random_nums.tolist()
        # ... other strategies ...

########### Generate Candidate ##############


    def _generate_candidate(self, strategy: str = None) -> List[int]:
        """Generate candidate numbers with time-based weighting integration.
        
        Args:
            strategy: Generation strategy ('balanced', 'frequent', or None for random)
            
        Returns:
            List of selected numbers (guaranteed native Python ints)
        """
        # Get base weights that already include time-based factors
        weights = self.weights.copy()
        
        # Apply strategy-specific time boosts if prediction is enabled
        if (strategy == 'aggressive' and 
            self.config.get('prediction', {}).get('enabled', False)):
            recent_window = self.config['prediction'].get('recent_time_window', 7)
            recent_weights = pd.Series(self.get_time_weights(window=recent_window))
            recent_ratio = self.config['prediction'].get('recent_time_ratio', 0.1)
            weights = weights * (1 - recent_ratio) + recent_weights * recent_ratio
            weights = weights / weights.sum()  # Renormalize
        
        if strategy == 'balanced':
            # Get hot and cold numbers (already time-weighted in base weights)
            hot = [int(n) for n in self.get_temperature_stats()['hot'][:3]]  # Convert to int
            cold = [int(n) for n in self.get_temperature_stats()['cold'][:2]]  # Convert to int
            remaining = self.config['strategy']['numbers_to_select'] - len(hot) - len(cold)
            
            # Filter pool and adjust weights for remaining numbers
            pool = [n for n in self.number_pool if n not in hot + cold]
            pool_weights = weights[pool]
            
            # Normalize weights for the remaining pool
            if pool_weights.sum() > 0:
                pool_weights = pool_weights / pool_weights.sum()
            else:  # Fallback if all weights are zero
                pool_weights = None
                
            random_nums = np.random.choice(
                pool,
                size=remaining,
                replace=False,
                p=pool_weights  # Use time-adjusted weights
            )
            return sorted(hot + cold + [int(n) for n in random_nums.tolist()])  # Convert all to int
            
        elif strategy == 'frequent':
            # Use weighted frequencies and convert to native ints
            return [int(n) for n in weights.nlargest(
                self.config['strategy']['numbers_to_select']
            ).index.tolist()]
            
        else:  # Fallback strategy with time weighting
            return sorted([int(n) for n in np.random.choice(
                self.number_pool,
                size=self.config['strategy']['numbers_to_select'],
                replace=False,
                p=weights  # Use time-adjusted weights
            )])

#############################################


    def _generate_fallback_set(self) -> List[int]:
        """Fallback if sum validation fails too often."""
        return sorted(np.random.choice(
            self.number_pool,
            size=self.config['strategy']['numbers_to_select'],
            replace=False
        ))

#===================
# end set generator
#===================

    def save_results(self, sets: Union[List[List[int]], List[Dict]], set_type: str = "optimized") -> str:
        """Save number sets to CSV with columns: numbers, sum, type, timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.config['data']['results_dir']) / f"sets_{timestamp}.csv"
        
        # Prepare data - handles both formats
        data = []
        for s in sets:
            if isinstance(s, dict):  # Optimized set (Dict format)
                numbers = s['numbers']
                current_sum = s.get('sum', sum(numbers))
            else:  # Raw set (List[int] format)
                numbers = s
                current_sum = sum(numbers)
                
            data.append({
                'numbers': '-'.join(map(str, numbers)),
                'sum': current_sum,
                'type': set_type,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        pd.DataFrame(data).to_csv(path, index=False)
        return str(path)

#==============================
# Mode Handler 
#===============================
################ Weights ################


    def _get_frequency_weights(self) -> pd.Series:
        """
        Convert historical frequencies to normalized weights (0-1 range).
        Returns:
            pd.Series: Weights indexed by number (e.g., {7: 0.035, 12: 0.028})
        """
        freqs = self.get_frequencies()
        if freqs.empty:
            return pd.Series(0, index=self.number_pool)
        return freqs / freqs.sum()  # Normalize

    def _get_recent_weights(self) -> pd.Series:
        """
        Weight numbers based on recency (hot/cold analysis).
        Returns:
            pd.Series: Weights with hot numbers boosted by 20%, cold penalized by 20%
        """
        temp_stats = self.get_temperature_stats()
        weights = pd.Series(1.0, index=self.number_pool)  # Default weight = 1.0
        
        # Apply hot/cold adjustments from config
        hot_boost = self.config['analysis'].get('hot_boost', 1.2)
        cold_penalty = self.config['analysis'].get('cold_penalty', 0.8)
        
        weights.loc[temp_stats['hot']] *= hot_boost
        weights.loc[temp_stats['cold']] *= cold_penalty
        
        return weights / weights.sum()  # Normalize

    def _get_gap_weights(self) -> dict:
        """
        Convert gap frequencies to weights for number generation.
        Returns:
            dict: {gap_size: weight} (e.g., {7: 0.15, 11: 0.1})
        """
        if not self.config['analysis']['gap_analysis']['enabled']:
            return {}
        
        gap_data = self.analyze_gaps()
        total = sum(gap_data['frequency'].values())
        return {gap: count/total for gap, count in gap_data['frequency'].items()}

    def _get_overdue_weights(self) -> dict:
        """
        Get weight boosts for overdue numbers.
        Returns:
            dict: {number: boost_factor} (e.g., {19: 1.2, 23: 1.2})
        """
        if not self.config['analysis']['overdue_analysis']['enabled']:
            return {}
        
        boost = self.config['analysis']['overdue_analysis'].get('weight_influence', 1.2)
        return {num: boost for num in self._get_overdue_numbers()}


    def _get_prime_weights(self) -> dict:  
        """Boost primes by configured weight (default: +20%)."""  
        if not self.config['analysis']['primes']['enabled']:  
            return {}  
        return {  
            num: self.config['analysis']['primes'].get('prime_weight', 1.2)  
            for num in self._get_prime_numbers()  
        }  


    def _get_odd_even_weights(self) -> dict:  
        """Penalize overrepresented odds/evens softly."""  
        if not self.config['analysis']['patterns']['odd_even']['enabled']:  
            return {}  
        penalty = self.config['analysis']['patterns']['odd_even'].get('weight_penalty', 0.9)  
        boost = self.config['analysis']['patterns']['odd_even'].get('weight_boost', 1.1)  
        return {  
            num: penalty if num % 2 != 0 else boost  # Penalize odds, boost evens  
            for num in self.number_pool  
        }  


    def _get_sum_weights(self) -> dict:  
        """Penalize numbers pushing sums outside Q1-Q3 range."""  
        sum_stats = self.get_sum_stats()  
        if sum_stats.get('error'):  
            return {}  
        avg = sum_stats['average']  
        return {  
            num: 0.8 if num > avg * 1.15 else 1.0  # Customize thresholds as needed  
            for num in self.number_pool  
        }  


    def _get_draws_in_window(self, window: int) -> pd.DataFrame:
        """Get draws from the last N days/draws based on config."""
        unit = self.config['analysis']['recency_units']  # 'draws' or 'days'
        if unit == 'draws':
            return pd.read_sql(f"SELECT * FROM draws ORDER BY date DESC LIMIT {window}", self.conn)
        else:
            cutoff = datetime.now() - timedelta(days=window)
            return pd.read_sql("SELECT * FROM draws WHERE date >= ?", self.conn, params=(cutoff,))

    def get_time_weights(self, window: int = None) -> dict:
        """Enhanced time-based weights with proper type handling"""
        window = window or self.config['prediction'].get('rolling_window', 30)
        
        try:
            recent_draws = self._get_draws_in_window(window)
            counts = recent_draws[[f'n{i}' for i in range(1, 7)]].stack().value_counts()
            
            # Create weights with native Python types from the start
            weights = {int(num): float(count) for num, count in counts.items()}
            
            # Initialize all numbers with minimum weight (0.1)
            full_weights = {num: 0.1 for num in self.number_pool}
            
            # Update with actual counts while preserving types
            full_weights.update(weights)
            
            # Convert to Series for pandas operations
            weights_series = pd.Series(full_weights)
            
            # Normalize with smoothing
            total = weights_series.sum()
            normalized = (weights_series / total).clip(lower=0.01)
            normalized /= normalized.sum()  # Final normalization
            
            # Return with guaranteed native Python types
            return {int(k): float(v) for k, v in normalized.items()}
            
        except Exception as e:
            logging.warning(f"Time weights fallback: {str(e)}")
            uniform = 1.0 / len(self.number_pool)
            return {int(num): float(uniform) for num in self.number_pool}


    def get_cooccurrence_weights(self) -> dict:
        """Calculate how often numbers appear together (for all pairs)."""
        depth = self.config['prediction'].get('cooccurrence_depth', 3)
        cooccurrence = {}
        for num in self.number_pool:
            pairs = self.get_combinations(size=2)
            related = pairs[(pairs['n1'] == num) | (pairs['n2'] == num)]
            top_pairs = related.nlargest(depth, 'frequency')
            cooccurrence[num] = {
                'pairs': [p for p in top_pairs[['n1', 'n2']].values if p != num],
                'weights': top_pairs['frequency'].to_dict()
            }
        return cooccurrence

#########################################
    def _init_mode_handler(self):
        """Initialize mode and weights (now hybrid-aware)"""
        self.mode = self.config.get('mode', 'auto')
        if hasattr(self, '_init_weights_hybrid'):  # Check if hybrid exists
            self._init_weights()  # This now routes to hybrid or original
        else:
            self._init_weights()  # Pure fallback
        

    def _init_weights(self):
        """Initialize weights with time awareness while preserving hybrid analysis."""
        # Preserve hybrid analysis check
        if self.config['analysis']['overdue_analysis']['enabled']:
            self._init_weights_hybrid()  # Keep your existing hybrid logic
            return

        # Core weight initialization (modified to include time awareness)
        if self.mode == 'auto':
            # Start with uniform weights
            base_weights = pd.Series(1.0, index=self.number_pool)
            
            # Add time awareness if enabled
            if self.config.get('prediction', {}).get('enabled', False):
                time_window = self.config['prediction'].get('global_time_window', 30)
                time_weights = pd.Series(self._get_time_weights(window=time_window)).fillna(0)
                global_ratio = self.config['prediction'].get('global_time_ratio', 0.15)
                base_weights = base_weights * (1 - global_ratio) + time_weights * global_ratio
            
            self.weights = base_weights / base_weights.sum()
            self.learning_rate = self.config.get('auto', {}).get('learning_rate', 0.01)
            self.decay_factor = self.config.get('auto', {}).get('decay_factor', 0.97)
            
        else:  # Manual mode
            weights_config = self.config.get('manual', {}).get('strategy', {}).get('weighting', {})
            
            # Base weights with time awareness
            base_weights = (
                weights_config.get('frequency', 0.4) * self._get_frequency_weights() +
                weights_config.get('recency', 0.3) * self._get_recent_weights() +
                weights_config.get('randomness', 0.3) * np.random.rand(len(self.number_pool))
            )
            
            # Add time awareness if enabled
            if self.config.get('prediction', {}).get('enabled', False):
                time_window = self.config['prediction'].get('global_time_window', 30)
                time_weights = pd.Series(self._get_time_weights(window=time_window)).fillna(0)
                global_ratio = self.config['prediction'].get('global_time_ratio', 0.15)
                base_weights = base_weights * (1 - global_ratio) + time_weights * global_ratio
            
            # Apply cold number bonus (preserve existing logic)
            cold_bonus = weights_config.get('resurgence', 0.1)
            cold_nums = self.get_temperature_stats()['cold']
            base_weights[cold_nums] *= (1 + cold_bonus)
            
            self.weights = base_weights / base_weights.sum()

    
    def _init_weights_hybrid(self):
        """New hybrid weight calculation with gap + temperature support (non-destructive)"""
        try:
            # Base weights (same as original auto/manual logic)
            if self.mode == 'auto':
                base_weights = pd.Series(1.0, index=self.number_pool)
            else:
                weights_config = self.config.get('manual', {}).get('strategy', {}).get('weighting', {})
                base_weights = (
                    weights_config.get('frequency', 0.4) * self._get_frequency_weights() +
                    weights_config.get('recency', 0.3) * self._get_recent_weights() +
                    weights_config.get('randomness', 0.3) * np.random.rand(len(self.number_pool))
                )

            # Apply cold number bonus (original behavior)
            cold_nums = self.get_temperature_stats()['cold']
            cold_bonus = self.config['manual']['strategy']['weighting'].get('resurgence', 0.1)
            base_weights[cold_nums] *= (1 + cold_bonus)

            # Apply gap analysis (new behavior)
            if self.config['analysis']['overdue_analysis']['enabled']:
                overdue_nums = set(self._get_overdue_numbers()) - set(cold_nums)  # Avoid overlap
                overdue_boost = self.config['analysis']['overdue_analysis']['weight_influence']
                base_weights[list(overdue_nums)] *= (1 + overdue_boost)

            # Normalize (same as original)
            self.weights = base_weights / base_weights.sum()

        except Exception as e:
            logging.warning(f"Hybrid weight init failed: {e}. Falling back to original.")
            self._init_weights()  # Fallback to original

###################################
    def set_mode(self, mode: str):
        """Change modes dynamically"""
        valid_modes = ['auto', 'manual']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Choose from: {valid_modes}")
        self.mode = mode
        self._init_weights()


#==============================
#End mode handler
#=============================

    def get_prime_stats(self) -> Dict:
        """Enhanced prime stats with structured output"""
        try:
            draw_limit = max(1, self._get_analysis_draw_limit('primes', 500))
            hot_threshold = self.config['analysis']['primes'].get('hot_threshold', 5)
            
            # Existing query execution
            result = self.conn.execute(query).fetchone()
            
            # Get additional prime data
            prime_nums = self._get_prime_numbers()
            temp_stats = self.get_temperature_stats()
            
            return {
                # Standard structured format
                "analysis": "prime_stats",
                "metadata": {
                    "draws_analyzed": draw_limit,
                    "hot_threshold": hot_threshold,
                    "total_primes": len(prime_nums)
                },
                "stats": {
                    "avg_primes_per_draw": round(result[0], 2),
                    "pct_two_plus_primes": round(result[1], 1),
                    "hot_primes": [n for n in temp_stats['numbers']['hot'] 
                                  if n in prime_nums],
                    "cold_primes": [n for n in temp_stats['numbers']['cold'] 
                                   if n in prime_nums]
                },
                "all_primes": sorted(prime_nums),
                
                # Backward compatible format
                "legacy_format": {
                    'avg_primes': round(result[0], 2),
                    'pct_two_plus': round(result[1], 1),
                    'error': None
                }
            }

        except sqlite3.Error as e:
            error_msg = f"Prime stats failed: {str(e)}"
            logging.error(error_msg)
            return {
                "analysis": "prime_stats",
                "error": error_msg,
                "legacy_format": {
                    'error': 'Prime analysis unavailable'
                }
            }

    def _is_prime(self, n: int) -> bool:
        """Helper method to check if a number is prime"""
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    def _get_prime_numbers(self) -> List[int]:
        """NEW: Get all primes in number pool"""
        return [n for n in self.number_pool if self._is_prime(n)]

    def get_prime_temperature_stats(self) -> Dict[str, List[int]]:
        """Classify primes as hot/cold based on draw counts."""
        temp_stats = self.get_temperature_stats()
        primes = set(self._get_prime_numbers())
        return {
            'hot_primes': sorted(n for n in temp_stats['hot'] if n in primes),
            'cold_primes': sorted(n for n in temp_stats['cold'] if n in primes)
        }
#######Detect Patterns Update ####### 
############################# Display Gap Analysis ##################

    def display_gap_analysis(self, gap_data: dict) -> None:
        """Print formatted gap analysis results (matches your requested format exactly)."""
        if not gap_data:
            print("Gap Analysis: Disabled in config")
            return

        cfg = self.config['analysis']['gap_analysis']
        lookback = cfg['lookback_draws']
        
        print("\nGap Analysis")
        print(f"  - Analyzed last {lookback} draws")
        
        # Frequency
        if gap_data.get('frequency'):
            top_gaps = sorted(gap_data['frequency'].items(), key=lambda x: -x[1])[:3]
            gaps_str = " | ".join(f"{g}: {c}x ({c/lookback*100:.1f}%)" for g, c in top_gaps)
            print(f"  - Top Gaps by Frequency: {gaps_str}")
        
        # Overdue
        if gap_data.get('overdue'):
            threshold = cfg['auto_threshold'] if cfg['mode'] == 'auto' else cfg['manual_threshold']
            for gap in gap_data['overdue'][:1]:  # Show top 1 overdue gap
                print(f"  - Overdue Gaps: {gap['gap']}: {gap['draws_overdue']} draws overdue "
                      f"(avg every {gap['avg_frequency']:.1f} draws, threshold: {threshold}x)")
        
        # Clustering
        if cfg['track_consecutive'] and gap_data.get('clusters'):
            cluster_pct = gap_data['clusters']['small_gap_clusters'] / lookback * 100
            print(f"  - Clustering: {gap_data['clusters']['small_gap_clusters']} draws "
                  f"({cluster_pct:.1f}%) have 2+ small gaps (≤5)")
        
        # Recommendations
        print("  - Prioritize sets with gaps:", ", ".join(str(g[0]) for g in sorted(gap_data['frequency'].items(), key=lambda x: -x[1])[:3]))
        if gap_data.get('overdue'):
            print(f"  - Consider testing gap={gap_data['overdue'][0]['gap']} (overdue)")
        print(f"  - Avoid gaps > {cfg['max_gap_size']} (historically rare)")
 
#####################################################################
 ############################ DISPLAY OPTIMIZED SET ##################
    def display_optimized_sets(self, sets: List[Dict]):
        """Safe display of optimized sets with proper error handling"""
        if not sets:
            print("\n=== OPTIMIZED SETS ===\nNo valid sets generated")
            return

        print("\n=== OPTIMIZED SETS ===")
        
        # Get overdue numbers if analysis is enabled
        overdue_nums = []
        if self.config['analysis']['overdue_analysis']['enabled']:
            overdue_nums = self._get_overdue_numbers()
        
        for i, s in enumerate(sets, 1):
            try:
                # Safely extract numbers and sum
                numbers = s.get('numbers', [])
                sum_total = s.get('sum', 0)
                
                # Number range calculation - now storing the actual numbers
                low_max = self.config['strategy'].get('low_number_max', 10)
                mid_max = low_max * 2  # Dynamic mid-range based on config
                
                # Get numbers in each range
                low_nums = [n for n in numbers if n <= low_max]
                mid_nums = [n for n in numbers if low_max < n <= mid_max]
                high_nums = [n for n in numbers if n > mid_max]
                
                # Count numbers in each range
                low = len(low_nums)
                mid = len(mid_nums)
                high = len(high_nums)
                
                # Parse notes
                sum_note = next((n for n in s.get('notes', []) if 'Sum:' in n), "Sum: N/A")
                even_odd_note = next((n for n in s.get('notes', []) if 'Even/Odd' in n), None)
                
                # Temperature stats
                temp_stats = self.get_temperature_stats()
                hot_nums = [n for n in numbers if n in temp_stats['hot']]
                cold_nums = [n for n in numbers if n in temp_stats['cold']]
                
                # Overdue number analysis
                included_overdue = [n for n in numbers if n in overdue_nums]
                overdue_display = ""
                if overdue_nums:
                    overdue_display = f"   - Overdue Coverage: {len(included_overdue)}/{len(overdue_nums)}"
                    if included_overdue:
                        overdue_display += f" ({', '.join(map(str, included_overdue))})"
                
                # Display set information
                print(f"{i}. {'-'.join(map(str, numbers))}")
                
                # Sum information
                if '|' in sum_note:
                    print(f"   - Sum: {sum_total} {sum_note.split('|')[1].strip()}")
                else:
                    print(f"   - Sum: {sum_total} (Optimal range: {sum_note.split(':')[1].strip() if ':' in sum_note else 'N/A'})")
                
                # Pattern notes
                if even_odd_note:
                    print(f"   - {even_odd_note}")
                
                # Number characteristics
                print(f"   - Hot Numbers: {len(hot_nums)} ({', '.join(map(str, hot_nums)) if hot_nums else 'None'})")
                print(f"   - Cold Numbers: {len(cold_nums)} ({', '.join(map(str, cold_nums)) if cold_nums else 'None'})")
                
                # Overdue information (if enabled)
                if overdue_nums:
                    print(overdue_display)
                
                # Updated number range display with specific numbers
                print(f"   - Number Range: {low} Low ({', '.join(map(str, low_nums)) if low_nums else 'None'}), "
                      f"{mid} Mid ({', '.join(map(str, mid_nums)) if mid_nums else 'None'}), "
                      f"{high} High ({', '.join(map(str, high_nums)) if high_nums else 'None'})\n")
                
            except Exception as e:
                print(f"   - Error displaying set {i}: {str(e)}")
                continue
 #####################################################################


    def display_pattern_analysis(self, patterns: Dict):
        """Formatted output for pattern analysis"""
        print("\n=== PATTERN ANALYSIS ===")
        
        # Even/Odd
        if 'even_odd' in patterns:
            eo = patterns['even_odd']
            print("Even/Odd Distribution:")
            print(f"- Current Sets: {eo['balanced_sets']} balanced, {eo['total_sets']-eo['balanced_sets']} unbalanced")
            print(f"- Recent Draws: {round(eo['even_count']/(eo['total_sets']*6)*100)}% even ({eo['total_sets']}-draw average)")
            print(f"- Ideal Range: {eo['ideal_range'][0]}-{eo['ideal_range'][1]}% even numbers\n")

        # Primes
        if 'primes' in patterns:
            p = patterns['primes']
            print("Prime Numbers:")
            print(f"- Avg per set: {p['avg']} (Ideal {p['ideal'][0]}-{p['ideal'][1]})")
            print(f"- Hot Primes: {', '.join(map(str, p['hot'])) or 'None'}\n")

        # Consecutives
        if 'consecutives' in patterns:
            c = patterns['consecutives']
            print("Consecutives:")
            print(f"- {c['found']} pair {'(Below max ' + str(c['max_allowed']) if c['found'] <= c['max_allowed'] else '(WARNING: Exceeds limit)'}\n")

        # Digit Endings
        if 'digit_endings' in patterns:
            print("Digit Endings:")
            print(f"- {patterns['digit_endings']} ({'No repeats' if len(set(patterns['digit_endings'])) == 6 else 'Repeats found'})\n")

        # Sum and Ranges
        if 'sum_stats' in patterns:
            s = patterns['sum_stats']
            print("Sum:")
            print(f"- {s['average']} ({s['q1']}-{s['q3']} optimal range)\n")

        if 'number_ranges' in patterns:
            nr = patterns['number_ranges']
            print("Number Ranges:")
            print(f"- Low: {nr['counts']['low']}")
            print(f"- Mid: {nr['counts']['mid']}")
            print(f"- High: {nr['counts']['high']}")
###################### End pattern Display ###################
    def detect_patterns(self) -> Dict:
        """Enhanced pattern detection with all required metrics"""
        if not self.config.get('features', {}).get('enable_pattern_analysis', False):
            return {}

        try:
            limit = self.config.get('pattern_settings', {}).get('sample_size', 100)
            recent = pd.read_sql(f"""
                SELECT n1, n2, n3, n4, n5, n6 FROM draws
                ORDER BY date DESC
                LIMIT {limit}
            """, self.conn)
            
            # Initialize combined pattern tracking
            patterns = {
                # Existing metrics
                'consecutive': 0,
                'same_ending': 0,
                'all_even_odd': 0,
                'prime_count': [],
                'avg_primes': 0,
                
                # New enhanced metrics
                'even_odd': {
                    'balanced_sets': 0,
                    'total_sets': 0,
                    'even_count': 0,
                    'ideal_range': self.config['analysis']['patterns']['odd_even']['ideal_range']
                },
                'primes_enhanced': {
                    'hot': [],
                    'ideal': [1, 3]
                },
                'consecutives_max': 0,
                'digit_endings': [],
                'sum_stats': self.get_sum_stats(),
                'number_ranges': self.get_number_ranges_stats()
            }

            for _, row in recent.iterrows():
                nums = sorted(row.tolist())
                diffs = [nums[i+1] - nums[i] for i in range(5)]
                
                # Existing checks
                if any(d == 1 for d in diffs):
                    patterns['consecutive'] += 1
                    
                last_digits = [n % 10 for n in nums]
                if len(set(last_digits)) < 3:
                    patterns['same_ending'] += 1
                    
                if all(n % 2 == 0 for n in nums) or all(n % 2 == 1 for n in nums):
                    patterns['all_even_odd'] += 1
                    
                primes = [n for n in nums if self._is_prime(n)]
                patterns['prime_count'].append(len(primes))
                
                # New enhanced checks
                even_count = sum(1 for n in nums if n % 2 == 0)
                patterns['even_odd']['even_count'] += even_count
                patterns['even_odd']['total_sets'] += 1
                if patterns['even_odd']['ideal_range'][0] <= (even_count/len(nums))*100 <= patterns['even_odd']['ideal_range'][1]:
                    patterns['even_odd']['balanced_sets'] += 1
                    
                patterns['primes_enhanced']['hot'].extend(p for p in primes if p in self.get_temperature_stats()['hot'])
                patterns['consecutives_max'] = max(patterns['consecutives_max'], 
                                                 sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1))
                patterns['digit_endings'].append(''.join(str(n%10) for n in nums))

            # Final calculations
            total_draws = len(recent)
            if total_draws > 0:
                patterns['consecutive'] = (patterns['consecutive'] / total_draws) * 100
                patterns['same_ending'] = (patterns['same_ending'] / total_draws) * 100
                patterns['all_even_odd'] = (patterns['all_even_odd'] / total_draws) * 100
                patterns['avg_primes'] = np.mean(patterns['prime_count']) if patterns['prime_count'] else 0
                
                patterns['primes_enhanced']['hot'] = sorted(list(set(patterns['primes_enhanced']['hot'])))
                patterns['digit_endings'] = ' '.join(sorted(set(patterns['digit_endings'][0]))) if patterns['digit_endings'] else 'N/A'
            
            return patterns

        except Exception as e:
            logging.error(f"Pattern detection failed: {str(e)}")
            return {}

######

    def _get_prime_subsets(self, numbers: List[int]) -> List[int]:
        """Extract primes from any number list."""
        return [n for n in numbers if self._is_prime(n)]

    def _tag_prime_combos(self, combos: pd.DataFrame, size: int) -> pd.DataFrame:
        """Add '[All Primes]' tag to combos where all numbers are prime."""
        primes = set(self._get_prime_numbers())
        combos['is_prime_combo'] = combos[
            [f'n{i}' for i in range(1, size+1)]
        ].apply(lambda row: all(n in primes for n in row), axis=1)
        return combos

########################

    def get_number_ranges_stats(self) -> dict:
        """Three-way number range analysis (Low-Mid-High)"""
        try:
            cfg = self.config['analysis']['number_ranges']
            pool_size = self.config['strategy']['number_pool']
            
            # Auto-adjust ranges if configured
            if cfg.get('dynamic_ranges', False):
                low_max = pool_size // 3
                mid_max = 2 * (pool_size // 3)
            else:
                low_max = cfg['low_max']
                mid_max = cfg['mid_max']
            
            draw_limit = self._get_analysis_draw_limit('number_ranges', 500)
            
            query = f"""
            WITH recent_draws AS (
                SELECT * FROM draws ORDER BY date DESC LIMIT {draw_limit}
            ),
            range_flags AS (
                SELECT 
                    date,
                    -- Low numbers
                    CASE WHEN n1 <= {low_max} OR n2 <= {low_max} OR 
                              n3 <= {low_max} OR n4 <= {low_max} OR
                              n5 <= {low_max} OR n6 <= {low_max} 
                         THEN 1 ELSE 0 END as has_low,
                    -- Mid numbers
                    CASE WHEN (n1 > {low_max} AND n1 <= {mid_max}) OR 
                              (n2 > {low_max} AND n2 <= {mid_max}) OR
                              (n3 > {low_max} AND n3 <= {mid_max}) OR
                              (n4 > {low_max} AND n4 <= {mid_max}) OR
                              (n5 > {low_max} AND n5 <= {mid_max}) OR
                              (n6 > {low_max} AND n6 <= {mid_max})
                         THEN 1 ELSE 0 END as has_mid,
                    -- High numbers
                    CASE WHEN n1 > {mid_max} OR n2 > {mid_max} OR 
                              n3 > {mid_max} OR n4 > {mid_max} OR
                              n5 > {mid_max} OR n6 > {mid_max}
                         THEN 1 ELSE 0 END as has_high
                FROM recent_draws
            )
            SELECT 
                AVG(has_low) * 100 as pct_low,
                AVG(has_mid) * 100 as pct_mid,
                AVG(has_high) * 100 as pct_high,
                SUM(has_low) as low_draws,
                SUM(has_mid) as mid_draws,
                SUM(has_high) as high_draws,
                COUNT(*) as total_draws,
                {low_max} as low_max,
                {mid_max} as mid_max
            FROM range_flags
            """
            result = self.conn.execute(query).fetchone()
            
            return {
                'ranges': {
                    'low': f"1-{result[7]}",
                    'mid': f"{result[7]+1}-{result[8]}", 
                    'high': f"{result[8]+1}-{pool_size}"
                },
                'percentages': {
                    'low': round(result[0], 1),
                    'mid': round(result[1], 1),
                    'high': round(result[2], 1)
                },
                'counts': {
                    'low': result[3],
                    'mid': result[4],
                    'high': result[5]
                },
                'total_draws': result[6]
            }
            
        except Exception as e:
            logging.error(f"Range analysis failed: {str(e)}")
            return {'error': 'Range analysis failed'}

########################

    def get_combination_stats(self, size: int) -> Dict:
        """Enhanced version with structured output and existing features"""
        if not self.config.get('features', {}).get('enable_combo_stats', False):
            return {
                "analysis": "combination_stats",
                "status": "disabled_in_config",
                "size": size
            }

        try:
            combos = self.get_combinations(size, verbose=False)
            if combos.empty:
                return {
                    "analysis": "combination_stats",
                    "status": "no_data",
                    "size": size
                }

            # Core calculations (preserving your existing logic)
            total_possible = len(list(combinations(self.number_pool, size)))
            co_occurrence = defaultdict(int)
            
            for _, row in combos.iterrows():
                for i in range(1, size+1):
                    num = row[f'n{i}']
                    co_occurrence[num] += 1

            most_common_row = combos.iloc[0]
            most_common_numbers = [most_common_row[f'n{i}'] for i in range(1, size+1)]

            # Build enhanced output structure
            return {
                "analysis": "combination_stats",
                "size": size,
                "status": "success",
                "summary": {
                    "average_frequency": float(combos['frequency'].mean()),
                    "std_deviation": float(combos['frequency'].std()),
                    "coverage_pct": (len(combos) / total_possible * 100,
                    "total_possible": total_possible,
                    "observed": len(combos)
                },
                "most_common": {
                    "numbers": most_common_numbers,
                    "count": int(most_common_row['frequency']),
                    "pct_of_draws": (most_common_row['frequency']/self._get_draw_count())*100
                },
                "co_occurrence": [
                    {"number": num, "count": count}
                    for num, count in sorted(co_occurrence.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:5]
                ],
                "top_combinations": [
                    {
                        "numbers": [row[f'n{i}'] for i in range(1, size+1)],
                        "count": row['frequency'],
                        "pct_of_draws": (row['frequency']/self._get_draw_count())*100
                    }
                    for _, row in combos.head(5).iterrows()
                ]
            }

        except Exception as e:
            logging.warning(f"Combination stats failed for size {size}: {str(e)}")
            return {
                "analysis": "combination_stats",
                "status": "error",
                "error": str(e),
                "size": size
            }

############################# New Added Section ###########################

    def get_combined_stats(self):
        """Calculate cross-category statistical relationships"""
        if not self.config['analysis'].get('show_combined_stats', False):
            return None
            
        return {
            'hot_frequent': self._get_hot_frequent_overlap(),
            'pattern_corr': self._get_pattern_correlations(),
            'coverage': self._get_coverage_stats()
        }
###########################
    def _get_hot_frequent_overlap(self):
        """Calculate overlap between hot and frequent numbers"""
        try:
            # Get hot numbers
            temp_stats = self.get_temperature_stats()
            hot_nums = set(temp_stats.get('hot', []))
            
            # Get frequent numbers
            freq_series = self.get_frequencies(20)  # Get top 20 frequent numbers
            
            # Handle empty cases
            if not hot_nums or freq_series.empty:
                return {'overlap_pct': 0, 'freq_multiplier': 0}
                
            freq_nums = set(freq_series.index.tolist())
            overlap = hot_nums.intersection(freq_nums)
            
            hot_count = len(hot_nums) or 1  # Prevent division by zero
            freq_mean = freq_series.mean()
            
            # Calculate overlap statistics
            if overlap:
                hot_freq_mean = freq_series.loc[list(overlap)].mean()
                multiplier = round(hot_freq_mean/freq_mean, 1) if freq_mean else 0
            else:
                multiplier = 0
                
            return {
                'overlap_pct': round(len(overlap)/hot_count*100, 1),
                'freq_multiplier': multiplier
            }
        except Exception as e:
            logging.warning(f"Hot-frequent analysis failed: {str(e)}")
            return {'overlap_pct': 0, 'freq_multiplier': 0}
######################################
    def _get_pattern_correlations(self):
        """Calculate pattern relationships"""
        # Implement your pattern correlation logic here
        return {
            'hot_freq_pair_rate': 62,  # Example value
            'cold_pair_reduction': 28   # Example value
        }

    def _get_coverage_stats(self):
        """Calculate coverage statistics"""
        # Implement your coverage analysis here
        return {
            'pattern_coverage': 82,  # Example value
            'never_paired_pct': 41   # Example value
        }

###########################################################################

# ======================
    # COMBINED ANALYSIS
    # ======================
 
    def run_analyses(self) -> dict:
        """
        Run all configured analyses and return consolidated results.
        
        Returns:
            {
                'frequency': pd.Series,
                'gaps': {  # New section
                    'frequency': dict,
                    'overdue': list,
                    'clusters': dict
                },
                'primes': {'avg_primes': float, ...},
                'high_low': {'pct_with_low': float, ...},
                'overdue_analysis': {
                    'overdue': List[int], 
                    'stats': {
                        'avg_overdue': float,
                        'max_overdue': int,
                        ...
                    },
                    'distribution': dict
                },
                'metadata': {
                    'effective_draws': {
                        'primes': int,
                        'high_low': int,
                        'overdue_analysis': int,
                        'gap_analysis': int  # New
                    }
                }
            }
        """
        results = {
            'frequency': self.get_frequencies(),
            'primes': self.get_prime_stats(),
            'high_low': self.get_highlow_stats(),
            'metadata': {
                'effective_draws': {
                    'primes': self._get_analysis_draw_limit('primes', 500),
                    'high_low': self._get_analysis_draw_limit('high_low', 400),
                    'gap_analysis': self._get_analysis_draw_limit('gap_analysis', 500)
                }
            }
        }

        # Gap Analysis (new with formatted output)
        if self.config['analysis']['gap_analysis']['enabled']:
            results['gaps'] = self.analyze_gaps()
            self.display_gap_analysis(results['gaps'])  # <-- New formatted output
        
        # Overdue Analysis (existing unchanged)
        if self.config['analysis']['overdue_analysis']['enabled']:
            results['overdue_analysis'] = {
                'overdue': self.get_overdue_numbers(),
                'stats': self.get_overdue_stats(),
                'distribution': self.get_overdue_distribution()
            }
            results['metadata']['effective_draws']['overdue_analysis'] = \
                self._get_draw_count()
        
        return results

############ SUMMARY ANALYSIS ######################

    def get_sum_stats(self) -> dict:
        """SQLite-compatible sum statistics using approximate percentiles"""
        query = """
        WITH sums AS (
            SELECT (n1+n2+n3+n4+n5+n6) as total,
                   COUNT() OVER () as n
            FROM draws
        ),
        sorted AS (
            SELECT total, ROW_NUMBER() OVER (ORDER BY total) as row_num
            FROM sums
        )
        SELECT
            AVG(total) as avg_sum,
            MIN(total) as min_sum,
            MAX(total) as max_sum,
            (SELECT total FROM sorted WHERE row_num = CAST(n*0.25 AS INT)) as q1_sum,
            (SELECT total FROM sorted WHERE row_num = CAST(n*0.5 AS INT)) as median_sum,
            (SELECT total FROM sorted WHERE row_num = CAST(n*0.75 AS INT)) as q3_sum
        FROM sums
        LIMIT 1
        """
        try:
            result = self.conn.execute(query).fetchone()
            return {
                'average': round(result[0], 1),
                'min': result[1],
                'max': result[2],
                'q1': round(result[3], 1),
                'median': round(result[4], 1),
                'q3': round(result[5], 1)
            }
        except sqlite3.Error as e:
            logging.error(f"Sum stats failed: {str(e)}")
            return {'error': 'Sum analysis failed'}

    def get_sum_frequencies(self, bin_size: int = 10) -> dict:
        """SQLite-compatible sum frequency bins using CAST instead of FLOOR"""
        query = f"""
        WITH sums AS (
            SELECT (n1+n2+n3+n4+n5+n6) as total 
            FROM draws
        ),
        bins AS (
            SELECT 
                CAST(total/{bin_size} AS INT)*{bin_size} as lower_bound,
                COUNT(*) as frequency
            FROM sums
            GROUP BY CAST(total/{bin_size} AS INT)
        )
        SELECT 
            lower_bound,
            lower_bound+{bin_size}-1 as upper_bound,
            frequency
        FROM bins
        ORDER BY lower_bound
        """
        try:
            rows = self.conn.execute(query).fetchall()
            return {
                f"{lb}-{ub}": freq for lb, ub, freq in rows
            }
        except sqlite3.Error as e:
            logging.error(f"Sum frequency failed: {str(e)}")
            return {'error': 'Sum frequency analysis failed'}

############### OVERDUE ANALYSIS #####################################

    def simulate_overdue_thresholds(self):
        """Test different threshold values"""
        results = []
        for threshold in [1.3, 1.5, 1.7, 2.0]:
            self.config['analysis']['overdue_analysis']['auto_threshold'] = threshold
            self._initialize_overdue_analysis()
            overdue = self.get_overdue_numbers()
            results.append({
                'threshold': threshold,
                'count': len(overdue),
                'accuracy': self._test_overdue_accuracy(overdue)
            })
        return results

    def _test_overdue_accuracy(self, numbers: List[int]) -> float:
        """Check if overdue numbers actually appeared soon after"""
        query = """
        SELECT COUNT(*) FROM draws
        WHERE ? IN (n1,n2,n3,n4,n5,n6)
        AND date BETWEEN date(?) AND date(?, '+7 days')
        """
        hits = 0
        for num in numbers:
            last_seen = self.conn.execute(
                "SELECT last_seen_date FROM number_overdues WHERE number = ?", (num,)
            ).fetchone()[0]
            hits += self.conn.execute(query, (num, last_seen, last_seen)).fetchone()[0]
        return hits / len(numbers) if numbers else 0

    def get_overdue_trends(self, num: int, lookback=10) -> dict:
        """Calculate gap trend for a specific number"""
        query = """
        WITH appearances AS (
            SELECT date FROM draws
            WHERE ? IN (n1,n2,n3,n4,n5,n6)
            ORDER BY date DESC
            LIMIT ?
        ),
        gaps AS (
            SELECT 
                julianday(a.date) - julianday(b.date) as gap
            FROM appearances a
            JOIN appearances b ON a.date > b.date
            LIMIT ?
        )
        SELECT 
            AVG(gap),
            (MAX(gap) - MIN(gap)) / COUNT(*)
        FROM gaps
        """
        avg_overdue, trend = self.conn.execute(query, (num, lookback, lookback)).fetchone()
        return {
            'number': num,
            'current_overdue': self.conn.execute(
                "SELECT current_overdue FROM number_overdues WHERE number = ?", (num,)
            ).fetchone()[0],
            'trend_slope': round(trend, 2),
            'is_accelerating': trend > 0.5  # Custom threshold
        }

    def get_overdue_stats(self) -> dict:
        """Calculate comprehensive gap statistics"""
        query = """
        WITH overdue_stats AS (
            SELECT 
                number,
                current_overdue,
                avg_overdue,
                CAST((julianday('now') - julianday(last_seen_date)) AS INTEGER) as days_since_seen
            FROM number_overdues
        )
        SELECT 
            AVG(current_overdue) as avg_overdue,
            MIN(current_overdue) as min_overdue,
            MAX(current_overdue) as max_overdue,
            AVG(days_since_seen) as avg_days_since,
            SUM(CASE WHEN is_overdue THEN 1 ELSE 0 END) as overdue_count
        FROM overdue_stats
        """
        result = self.conn.execute(query).fetchone()
        return {
            'average_overdue': round(result[0], 1),
            'min_overdue': result[1],
            'max_overdue': result[2],
            'avg_days_since_seen': round(result[3], 1),
            'overdue_count': result[4]
        }

    def get_overdue_distribution(self, bin_size=5) -> dict:
        """Bin gaps into ranges for histogram"""
        query = f"""
        SELECT 
            (current_overdue / {bin_size}) * {bin_size} as lower_bound,
            COUNT(*) as frequency
        FROM number_overdues
        GROUP BY (current_overdue / {bin_size})
        ORDER BY lower_bound
        """
        return {
            f"{row[0]}-{row[0]+bin_size-1}": row[1] 
            for row in self.conn.execute(query).fetchall()
        }


    def get_overdue_numbers(self, enhanced: bool = False) -> Union[List[int], List[dict]]:
        """Get overdue numbers with optional enhanced analytics
        
        Args:
            enhanced: If True, returns list of dicts with trend analysis
            
        Returns:
            List[int] if enhanced=False (default)
            List[dict] if enhanced=True {number: int, current_overdue: int, trend_slope: float}
        """
        if not self.config['analysis']['overdue_analysis']['enabled']:
            return [] if not enhanced else [{}]
        
        query = """
        SELECT number FROM number_overdues 
        WHERE is_overdue = TRUE
        ORDER BY current_overdue DESC
        LIMIT ?
        """
        top_n = self.config['analysis']['top_range']
        numbers = [row[0] for row in self.conn.execute(query, (top_n,))]
        
        if enhanced:
            return [{
                'number': num,
                'current_overdue': self.conn.execute(
                    "SELECT current_overdue FROM number_overdues WHERE number = ?", (num,)
                ).fetchone()[0],
                'trend_slope': self.get_overdue_trends(num)['trend_slope']
            } for num in numbers]
        return numbers


    def _parse_date(self, date_str):
        """Flexible date parser handling multiple formats"""
        formats = [
            '%Y/%m/%d',    # YYYY/MM/DD
            '%Y-%m-%d %H:%M:%S',  # YYYY-MM-DD HH:MM:SS
            '%Y-%m-%d',    # YYYY-MM-DD
            '%m/%d/%Y',    # MM/DD/YYYY (fallback)
            '%m/%d/%y'     # MM/DD/YY (fallback)
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Date '{date_str}' doesn't match any expected formats")

    def _initialize_overdue_analysis(self):
        if not self.config['analysis']['overdue_analysis']['enabled']:
            return
        if self.config['output'].get('verbose', True):    
            print("\nINITIALIZING GAP ANALYSIS...")

        # 1. Get all numbers that have ever appeared
        existing_nums = set()
        for i in range(1,7):
            nums = self.conn.execute(f"SELECT DISTINCT n{i} FROM draws").fetchall()
            existing_nums.update(n[0] for n in nums)
        
        # 2. Initialize table with ALL pool numbers
        self.conn.executemany(
            "INSERT OR IGNORE INTO number_overdues (number) VALUES (?)",
            [(n,) for n in self.number_pool]
        )
        
        # 3. Calculate initial gaps for existing numbers
        for num in existing_nums:
            # Get all appearance dates for this number
            dates = self.conn.execute("""
                SELECT date FROM draws
                WHERE ? IN (n1,n2,n3,n4,n5,n6)
                ORDER BY date
            """, (num,)).fetchall()
            
            if not dates:
                continue
                
            # Convert dates using flexible parser
            date_objs = []
            for d in dates:
                try:
                    date_objs.append(self._parse_date(d[0]))
                except ValueError as e:
                    print(f"⚠️ Failed to parse date {d[0]} for number {num}: {e}")
                    continue
                    
            if not date_objs:
                continue
                
            last_seen = dates[-1][0]  # Keep original string for DB storage
            
            # Calculate current gap
            latest_date_str = self.conn.execute(
                "SELECT MAX(date) FROM draws"
            ).fetchone()[0]
            try:
                latest_date = self._parse_date(latest_date_str)
                current_overdue = (latest_date - date_objs[-1]).days
            except ValueError as e:
                print(f"⚠️ Failed to calculate gap for number {num}: {e}")
                continue
            
            # Calculate historical average gap
            if len(date_objs) > 1:
                overdues = [(date_objs[i+1] - date_objs[i]).days 
                       for i in range(len(date_objs)-1)]
                avg_overdue = sum(overdues) / len(overdues)
            else:
                avg_overdue = current_overdue
                
            # Determine overdue status
            mode = self.config['analysis']['overdue_analysis']['mode']
            auto_thresh = self.config['analysis']['overdue_analysis']['auto_threshold']
            manual_thresh = self.config['analysis']['overdue_analysis']['manual_threshold']
            
            is_overdue = (current_overdue >= manual_thresh) if mode == 'manual' else (
                         current_overdue >= avg_overdue * auto_thresh)
            
            # Update record
            self.conn.execute("""
                UPDATE number_overdues
                SET last_seen_date = ?,
                    current_overdue = ?,
                    avg_overdue = ?,
                    is_overdue = ?
                WHERE number = ?
            """, (last_seen, current_overdue, avg_overdue, int(is_overdue), num))
        
        self._verify_overdue_analysis()

###############
    def update_overdue_stats(self):
        """Update gap statistics after new draws"""
        if not self.config['analysis']['overdue_analysis']['enabled']:
            return
            
        # Get latest draw date and numbers
        latest = self.conn.execute(
            "SELECT date, n1, n2, n3, n4, n5, n6 FROM draws ORDER BY date DESC LIMIT 1"
        ).fetchone()
        
        if not latest:
            return
            
        latest_date, *latest_nums = latest
        
        # Update gaps for all numbers
        self.conn.execute("""
            UPDATE number_overdues 
            SET current_overdue = current_overdue + 1,
                is_overdue = CASE
                    WHEN ? = 'manual' THEN current_overdue + 1 >= ?
                    ELSE current_overdue + 1 >= avg_overdue * ?
                END
        """, (
            self.config['analysis']['overdue_analysis']['mode'],
            self.config['analysis']['overdue_analysis']['manual_threshold'],
            self.config['analysis']['overdue_analysis']['auto_threshold']
        ))
        
        # Reset gaps for numbers in latest draw
        self.conn.executemany("""
            UPDATE number_overdues 
            SET last_seen_date = ?,
                current_overdue = 0,
                is_overdue = FALSE
            WHERE number = ?
        """, [(latest_date, num) for num in latest_nums])
        
        # Recalculate average gaps
        self._recalculate_avg_overdues()

###########################################################################
########################

#### ANALYSIS OUTPUT ###########

class AnalysisOutput:
    @staticmethod
    def to_json(analyzer, numbers_to_analyze=None):
        """Main structured output generator"""
        if numbers_to_analyze is None:
            numbers_to_analyze = analyzer.get_frequencies().index.tolist()[:5]
        
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "number_pool": analyzer.config['strategy']['number_pool'],
                "draw_count": analyzer._get_draw_count()
            },
            "analyses": [
                analyzer.get_temperature_stats(),
                analyzer.get_prime_stats(),
                *[analyzer.get_number_trend(n) for n in numbers_to_analyze],
                analyzer.get_combination_stats(2),
                analyzer.get_combination_stats(3),
                analyzer.get_overdue_trends()  # You'll need to implement this similarly
            ]
        }

    @staticmethod
    def to_cli(analyzer):
        """Simplified CLI output"""
        data = AnalysisOutput.to_json(analyzer)
        print(f"Analysis complete for {data['metadata']['draw_count']} draws")
        print(f"Number pool: 1-{data['metadata']['number_pool']}")
        
        for analysis in data['analyses']:
            print(f"\n{analysis['analysis'].upper().replace('_', ' ')}:")
            if analysis['analysis'] == 'number_trend':
                print(f"Number {analysis['number']}: "
                      f"{analysis['appearance_count']} appearances, "
                      f"current gap {analysis['current_gap']} draws")

################################

# ======================
# DASHBOARD GENERATOR
# ======================

class DashboardGenerator:
    def __init__(self, analyzer: LotteryAnalyzer):
        self.analyzer = analyzer
        self.dashboard_dir = Path(analyzer.config['data']['results_dir']) / "dashboard"
        self.dashboard_dir.mkdir(exist_ok=True)

    # ======================
    # NEW SAFE PARSER METHOD
    # ======================
    def _parse_combo_size(self, size_name: str) -> int:
        """Convert 'triplets' → 3 with validation"""
        size_map = {
            'pairs': 2,
            'triplets': 3,
            'quadruplets': 4,
            'quintuplets': 5,
            'sixtuplets': 6
        }
        size_name = size_name.lower().strip()
        if size_name not in size_map:
            raise ValueError(
                f"Invalid combo size '{size_name}'. "
                f"Must be one of: {list(size_map.keys())}"
            )
        return size_map[size_name]

    def _generate_number_card(self, title: str, numbers: list, color_class: str) -> str:
        """Generate a card with number bubbles"""
        numbers_html = "".join(
            f'<div class="number-bubble {color_class}">{num}</div>'
            for num in numbers[:15]  # Show up to 15 numbers
        )
        return f"""
        <div class="analysis-card">
            <h3>{title}</h3>
            <div class="number-grid">{numbers_html}</div>
        </div>
        """
#==============
#New Chart 
#==============

    def _generate_combination_chart(self, size: int) -> str:
        """Generate combination frequency chart"""
        top_n = self.analyzer.config['analysis']['top_range']
        combos = self.analyzer.get_combinations(size)
        
        labels = [f"{'-'.join(map(str, row[:-1]))}" for _, row in combos.iterrows()]
        counts = combos['frequency'].tolist()
        
        return f"""
        <div class="chart-card">
            <h3>Top {top_n} {size}-Number Combinations</h3>
            <div class="chart-container">
                <canvas id="comboChart{size}"></canvas>
            </div>
            <div class="chart-data" hidden>
                {json.dumps({"combinations": labels, "counts": counts})}
            </div>
        </div>
        """

#===================
    def _generate_frequency_chart(self, frequencies: pd.Series) -> str:
        """Generate the frequency chart HTML"""
        top_n = self.analyzer.config['analysis']['top_range']  # USE CONFIG VALUE
        top_numbers = frequencies.head(top_n).index.tolist()
        counts = frequencies.head(top_n).values.tolist()
        
        return f"""
        <div class="chart-card">
            <h3>Top {top_n} Frequent Numbers</h3>  <!-- DYNAMIC TITLE -->
            <div class="chart-container">
                <canvas id="frequencyChart"></canvas>
            </div>
            <div class="chart-data" hidden>
                {json.dumps({"numbers": top_numbers, "counts": counts})}
            </div>
        </div>
        """

    def _generate_recent_draws(self, count: int = 5) -> str:
        """Show recent draws"""
        recent = pd.read_sql(
            f"SELECT * FROM draws ORDER BY date DESC LIMIT {count}",
            self.analyzer.conn
        )
        rows = "".join(
            f"<tr><td>{row['date']}</td><td>{'-'.join(str(row[f'n{i}']) for i in range(1,7))}</td></tr>"
            for _, row in recent.iterrows()
        )
        return f"""
        <div class="recent-card">
            <h3>Last {count} Draws</h3>
            <table>
                <tr><th>Date</th><th>Numbers</th></tr>
                {rows}
            </table>
        </div>
        """

    def generate(self) -> str:
        """Generate complete dashboard"""
        # Get analysis data
        freqs = self.analyzer.get_frequencies()
        temps = self.analyzer.get_temperature_stats()
        
        # Generate HTML components
        cards = [
            self._generate_number_card("Hot Numbers", temps['hot'], 'hot'),
            self._generate_number_card("Cold Numbers", temps['cold'], 'cold'),
            self._generate_number_card("Frequent Numbers", freqs.index.tolist(), 'frequent'),
            self._generate_frequency_chart(freqs),
            self._generate_recent_draws()
        ]
        # ADD THIS NEW BLOCK FOR COMBINATION CHARTS
        combo_config = self.analyzer.config['analysis'].get('combination_analysis', {})
        for size_name, enabled in combo_config.items():
            try:
                if enabled:
                    size_num = self._parse_combo_size(size_name)  # Use the safe parser
                    cards.append(self._generate_combination_chart(size_num))
            except (ValueError, KeyError) as e:
                import logging
                logging.warning(f"Skipping invalid combo size '{size_name}': {str(e)}")
                
                
        # Complete HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lottery Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                }}
                .analysis-card, .chart-card, .recent-card {{
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 15px;
                }}
                .number-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(30px, 1fr));
                    gap: 8px;
                    margin-top: 10px;
                }}
                .number-bubble {{
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                }}
                .hot {{ background-color: #ff6b6b; color: white; }}
                .cold {{ background-color: #74b9ff; color: white; }}
                .frequent {{ background-color: #2ecc71; color: white; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                .chart-container {{ position: relative; height: 300px; margin-top: 15px; }}
                h3 {{ margin-top: 0; color: #2c3e50; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>Lottery Analysis Dashboard</h1>
            <div class="dashboard">
                {"".join(cards)}
            </div>
            <script>
                // Initialize frequency chart
                const freqData = JSON.parse(
                    document.querySelector('.chart-data').innerHTML
                );
                new Chart(
                    document.getElementById('frequencyChart'),
                    {{
                        type: 'bar',
                        data: {{
                            labels: freqData.numbers,
                            datasets: [{{
                                label: 'Appearances',
                                data: freqData.counts,
                                backgroundColor: '#2ecc71'
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false
                        }}
                    }}
                );
            </script>
        </body>
        </html>
        """
        
        # Save to file
        (self.dashboard_dir / "index.html").write_text(html)
        return str(self.dashboard_dir / "index.html")

# ======================
# MAIN APPLICATION
# ======================
def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load YAML config with defaults"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Deep merge
        def merge(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    merge(d1[k], v)
                else:
                    d1[k] = v
        merged = DEFAULT_CONFIG.copy()
        merge(merged, config)
        
        if 'features' in config and 'enable_pattern_analysis' in config['features']:
            config['analysis']['patterns']['enabled'] = config['features']['enable_pattern_analysis']
        
        return merged
    except Exception:
        return DEFAULT_CONFIG

def main():

    parser = argparse.ArgumentParser(description='Lottery Number Optimizer')
    parser.add_argument('--mode', choices=['auto', 'manual'], 
                       help='Override config mode setting')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--strategy', default='balanced', choices=['balanced', 'frequent'])
    parser.add_argument('--no-dashboard', action='store_true', help='Disable dashboard generation')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')

    # ============ INSERT NEW ARGUMENTS HERE ============
    parser.add_argument('--show-combos', nargs='+', 
                       choices=['pairs', 'triplets', 'quadruplets', 'quintuplets', 'sixtuplets'],
                       help="Override config to show specific combinations (e.g., --show-combos pairs triplets)")
    parser.add_argument('--hide-combos', nargs='+',
                       choices=['pairs', 'triplets', 'quadruplets', 'quintuplets', 'sixtuplets'],
                       help="Override config to hide specific combinations")
    parser.add_argument('--show-patterns', action='store_true',
                       help='Enable pattern detection analysis')
    parser.add_argument('--show-stats', action='store_true',
                       help='Enable combination statistics')
    # ============ END OF NEW ARGUMENTS ============

    args = parser.parse_args()



    
    try:
        # Initialize analyzer
        config = load_config(args.config)
#================
# New section
#================

        # Apply CLI overrides to combination analysis
        if args.show_combos or args.hide_combos:
            # Initialize if missing
            config['analysis']['combination_analysis'] = config.get('analysis', {}).get('combination_analysis', {})
            
            # Set all to False first if --show-combos is used (to enforce exclusivity)
            if args.show_combos:
                for combo in ['pairs', 'triplets', 'quadruplets', 'quintuplets', 'sixtuplets']:
                    config['analysis']['combination_analysis'][combo] = False
            
            # Apply CLI selections
            for combo in (args.show_combos or []):
                config['analysis']['combination_analysis'][combo] = True
            for combo in (args.hide_combos or []):
                config['analysis']['combination_analysis'][combo] = False

#================
        
        analyzer = LotteryAnalyzer(config)
        # Load and validate data
        analyzer.load_data()
        #if analyzer.config['analysis']['overdue_analysis']['enabled']:
        #    analyzer.debug_overdue_status()
        # Get analysis results
        freqs = analyzer.get_frequencies()
        temps = analyzer.get_temperature_stats()
        sets = analyzer.generate_sets(args.strategy)
        top_range = analyzer.config['analysis']['top_range']
        
        # Console output (unless --quiet)
        
        # ============ INSERT FEATURE RESULTS INIT HERE ============
        # Initialize feature_results dictionary
        feature_results = {
            'patterns': analyzer.detect_patterns() if (args.show_patterns or 
                     config.get('features', {}).get('enable_pattern_analysis', False)) else None,
            'stats': {
                2: analyzer.get_combination_stats(2) if (args.show_stats or 
                    config.get('features', {}).get('enable_combo_stats', False)) else None,
                3: analyzer.get_combination_stats(3) if (args.show_stats or 
                    config.get('features', {}).get('enable_combo_stats', False)) else None
            }
        }
        # ============ END FEATURE RESULTS INIT ============
        temp_stats = analyzer.get_temperature_stats()
        prime_temp_stats = analyzer.get_prime_temperature_stats()
        overdue = analyzer.get_overdue_numbers() 
        if not args.quiet:
            print("\n" + "="*50)
            print(" LOTTERY ANALYSIS RESULTS ".center(50, "="))
            print(f"\n Top {top_range} Frequent Numbers:")
            print(freqs.to_string())
            
            print(f"\nHot Numbers (last {config['analysis']['recency_bins']['hot']} draws):")
            print(f"   Numbers: {', '.join(map(str, temp_stats['hot']))}")
            print(f"   Primes: {', '.join(map(str, prime_temp_stats['hot_primes'])) or 'None'}")

            print(f"\n️Cold Numbers ({config['analysis']['recency_bins']['cold']}+ draws unseen):")
            print(f"   Numbers: {', '.join(map(str, temp_stats['cold']))}")
            print(f"   Primes: {', '.join(map(str, prime_temp_stats['cold_primes'])) or 'None'}")


        # New Overdue Numbers Section
        if overdue:
            print(f"\nOverdue Numbers ({config['analysis']['overdue_analysis']['manual_threshold']}+ draws unseen):")
            print(f"   Tracking: {len(analyzer._get_overdue_numbers())} total")
            print(f"   Numbers: {', '.join(map(str, overdue))}")
            
            # Prime number display
            overdue_primes = [n for n in overdue if analyzer._is_prime(n)]
            print(f"   Primes: {', '.join(map(str, overdue_primes)) if overdue_primes else 'None'}")

########################## GAP ANALYSIS #####################

### ========== INSERT GAP ANALYSIS OUTPUT HERE ========== ###
        if not args.quiet and analyzer.config['analysis']['gap_analysis']['enabled']:
            gap_results = analyzer.analyze_gaps()
            print("\nGap Analysis")
            print(f"  - Analyzed last {analyzer.config['analysis']['gap_analysis']['lookback_draws']} draws")
            
            # Frequency - Single line format
            if gap_results.get('frequency'):
                top_gaps = sorted(gap_results['frequency'].items(), key=lambda x: -x[1])[:3]
                gaps_str = " | ".join(f"{g}: {c}x ({c/analyzer.config['analysis']['gap_analysis']['lookback_draws']*100:.1f}%)" 
                                    for g, c in top_gaps)
                print(f"  - Top Gaps by Frequency: {gaps_str}")
            
            # Overdue - With threshold
            if gap_results.get('overdue'):
                threshold = (analyzer.config['analysis']['gap_analysis']['auto_threshold'] 
                           if analyzer.config['analysis']['gap_analysis']['mode'] == 'auto' 
                           else analyzer.config['analysis']['gap_analysis']['manual_threshold'])
                for gap in gap_results['overdue'][:1]:  # Just show top overdue gap
                    print(f"  - Overdue Gaps: {gap['gap']}: {gap['draws_overdue']} draws overdue "
                         f"(avg every {gap['avg_frequency']:.1f} draws, threshold: {threshold}x)")
            
            # Clustering
            if (analyzer.config['analysis']['gap_analysis']['track_consecutive'] and 
                gap_results.get('clusters')):
                cluster_pct = (gap_results['clusters']['small_gap_clusters'] / 
                              analyzer.config['analysis']['gap_analysis']['lookback_draws']) * 100
                print(f"  - Clustering: {gap_results['clusters']['small_gap_clusters']} draws "
                     f"({cluster_pct:.1f}%) have 2+ small gaps (≤5)")
            
            # Recommendations
            print("  - Prioritize sets with gaps:", 
                 ", ".join(str(g[0]) for g in sorted(gap_results['frequency'].items(), key=lambda x: -x[1])[:3]))
            if gap_results.get('overdue'):
                print(f"  - Consider testing gap={gap_results['overdue'][0]['gap']} (overdue)")
            print(f"  - Avoid gaps > {analyzer.config['analysis']['gap_analysis']['max_gap_size']} (historically rare)")
### ========== END GAP ANALYSIS OUTPUT ========== ###

#############################################################

######## HIGH LOW ###############

        range_stats = analyzer.get_number_ranges_stats()
        if not range_stats.get('error'):
            print(f"\nNumber Ranges Analysis:")
            print(f"   - Low ({range_stats['ranges']['low']}): {range_stats['percentages']['low']}% ({range_stats['counts']['low']} draws)")
            print(f"   - Mid ({range_stats['ranges']['mid']}): {range_stats['percentages']['mid']}% ({range_stats['counts']['mid']} draws)") 
            print(f"   - High ({range_stats['ranges']['high']}): {range_stats['percentages']['high']}% ({range_stats['counts']['high']} draws)")
            print(f"   Total analyzed: {range_stats['total_draws']} draws")
##############################################
#==================
# New Section
#==================
            print("\nTop Combinations:")
            combo_config = analyzer.config['analysis']['combination_analysis']
            
            for size, size_name in [(2, 'pairs'), (3, 'triplets'), 
                                 (4, 'quadruplets'), (5, 'quintuplets'), 
                                 (6, 'sixtuplets')]:
                if combo_config.get(size_name, False):
                    combos = analyzer.get_combinations(size)
                    if not combos.empty:
                        print(f"\nTop {len(combos)} {size_name}:")
                        for _, row in combos.iterrows():
                            nums = [str(row[f'n{i}']) for i in range(1, size+1)]
                            print(f"- {'-'.join(nums)} (appeared {row['frequency']} times)")
                        combos = analyzer._tag_prime_combos(combos, size)

            # ============ INSERT NEW FEATURE OUTPUTS HERE ============
##### Pattern Analysis ####
###########################

            if feature_results['stats'][2] or feature_results['stats'][3]:
                print("\n" + "="*50)
                print(" COMBINATION STATISTICS ".center(50, "="))
                for size in [2, 3]:
                    if feature_results['stats'][size]:
                        stats = feature_results['stats'][size]
                        print(f"\n▶ {size}-Number Combinations:")
                        print(f"  Average appearances: {stats['average_frequency']:.1f}")
                        print(f"  Most frequent: {'-'.join(map(str, stats['most_common']['numbers']))} "
                            f"(appeared {stats['most_common']['count']} times)")
            # ============ END NEW OUTPUTS ============
            if config['analysis'].get('show_combined_stats', False):
                combined = analyzer.get_combined_stats()
                if combined:
                    print("\n" + "="*50)
                    print(" COMBINED STATISTICAL INSIGHTS ".center(50, "="))
                    
                    hf = combined['hot_frequent']
                    print(f"\n● Hot & Frequent Numbers:")
                    print(f"   - {hf['overlap_pct']}% of hot numbers are also top frequent")
                    print(f"   - Appear {hf['freq_multiplier']}x more often than average")
                    
            # In main(), after other analyses:
            sum_stats = analyzer.get_sum_stats()
            sum_freq = analyzer.get_sum_frequencies()

            if not args.quiet and not sum_stats.get('error'):
                print("\nSum Range Analysis:")
                print(f"   Historical average: {sum_stats['average']}")
                print(f"   Q1-Q3 range: {sum_stats['q1']}-{sum_stats['q3']}")
                print(f"   Min-Max: {sum_stats['min']}-{sum_stats['max']}")
                
                print("\nCommon Sum Ranges:")
                for rng, freq in sorted(sum_freq.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"   {rng}: {freq} draws")
                    
#==================
            print("\nRecommended Number Sets:")
            for i, nums in enumerate(sets, 1):
                print(f"Set {i}: {'-'.join(map(str, nums))}")
            print("\n" + "="*50)
            #################################

            valid_sets = analyzer.generate_valid_sets()

            # NEW INTEGRATION POINT (replaces deleted code)
            patterns = analyzer.detect_patterns()
            if not args.quiet:
                analyzer.display_pattern_analysis(patterns)
                analyzer.display_optimized_sets(valid_sets)

        # Save files
            if config['output'].get('verbose', True): 
                print("Balanced:", analyzer._generate_candidate('balanced'))  # Uses base_weights + manual picks
                print("Aggressive:", analyzer._generate_candidate('aggressive'))  # Extra 10% recent boost
                print("Frequent:", analyzer._generate_candidate('frequent'))  # Pure frequency-based              
######################### FILE SAVING ########## #################################
        # Generate and save raw sets (pre-optimization)
# In your main() function, replace the saving section with:

            # Generate both sets
            raw_sets = analyzer.generate_sets(args.strategy)  # List[List[int]]
            optimized_sets = analyzer.generate_valid_sets()   # List[Dict]

            # Combine into one DataFrame
            all_sets = []
            for s in raw_sets:
                all_sets.append({
                    'numbers': '-'.join(map(str, s)),
                    'sum': sum(s),
                    'type': 'raw',
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            for s in optimized_sets:
                all_sets.append({
                    'numbers': '-'.join(map(str, s['numbers'])),
                    'sum': s.get('sum', sum(s['numbers'])),
                    'type': 'optimized', 
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            # Save combined results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(analyzer.config['data']['results_dir']) / f"sets_{timestamp}.csv"
            pd.DataFrame(all_sets).to_csv(path, index=False)

            if not args.quiet:
                print(f"\nSaved {len(raw_sets)} raw + {len(optimized_sets)} optimized sets to: {path}")
##################################################################################
        # Generate dashboard (unless --no-dashboard)
        if not args.no_dashboard:
            dashboard = DashboardGenerator(analyzer)
            dashboard_path = dashboard.generate()
            if not args.quiet:
                print(f"Dashboard generated at: {dashboard_path}")
                print("   View with: python -m http.server --directory results/dashboard 8000")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print(f"1. Check {args.config} exists and is valid")
        print(f"2. Verify data/numbers are 1-{config.get('strategy',{}).get('number_pool',55)}")
        print("3. Ensure CSV format: date,n1-n2-n3-n4-n5-n6")

if __name__ == "__main__":
    main()