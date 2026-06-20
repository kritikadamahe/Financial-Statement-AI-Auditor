import os
import logging
import random
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MonteCarloConfig")

DEFAULT_CONFIG = {
    "seed": 42,
    "num_normal": 1000,
    "num_fraud": 200,
    "start_year": 2020,
    "end_year": 2024,
    "pareto_alpha": 1.15,
    "sectors": {
        "Technology": {
            "base_revenue_min": 500000.0,
            "revenue_drift": 0.12,
            "revenue_volatility": 0.15,
            "cogs_ratio_mean": 0.40,
            "cogs_ratio_std": 0.05,
            "opex_ratio_mean": 0.25,
            "opex_ratio_std": 0.04,
            "asset_turnover_mean": 1.5,
            "asset_turnover_std": 0.2,
            "leverage_mean": 0.4,
            "leverage_std": 0.1,
            "dso_mean": 45.0,
            "dso_std": 5.0,
            "dio_mean": 30.0,
            "dio_std": 4.0,
            "dpo_mean": 40.0,
            "dpo_std": 5.0,
            "depreciation_rate_mean": 0.06,
            "depreciation_rate_std": 0.01
        },
        "Manufacturing": {
            "base_revenue_min": 2000000.0,
            "revenue_drift": 0.05,
            "revenue_volatility": 0.08,
            "cogs_ratio_mean": 0.60,
            "cogs_ratio_std": 0.04,
            "opex_ratio_mean": 0.20,
            "opex_ratio_std": 0.03,
            "asset_turnover_mean": 1.0,
            "asset_turnover_std": 0.1,
            "leverage_mean": 0.8,
            "leverage_std": 0.15,
            "dso_mean": 60.0,
            "dso_std": 6.0,
            "dio_mean": 75.0,
            "dio_std": 8.0,
            "dpo_mean": 50.0,
            "dpo_std": 5.0,
            "depreciation_rate_mean": 0.08,
            "depreciation_rate_std": 0.01
        },
        "Retail": {
            "base_revenue_min": 3000000.0,
            "revenue_drift": 0.08,
            "revenue_volatility": 0.10,
            "cogs_ratio_mean": 0.65,
            "cogs_ratio_std": 0.05,
            "opex_ratio_mean": 0.22,
            "opex_ratio_std": 0.03,
            "asset_turnover_mean": 2.0,
            "asset_turnover_std": 0.2,
            "leverage_mean": 0.6,
            "leverage_std": 0.12,
            "dso_mean": 15.0,
            "dso_std": 2.0,
            "dio_mean": 90.0,
            "dio_std": 9.0,
            "dpo_mean": 60.0,
            "dpo_std": 6.0,
            "depreciation_rate_mean": 0.05,
            "depreciation_rate_std": 0.01
        },
        "Healthcare": {
            "base_revenue_min": 1000000.0,
            "revenue_drift": 0.07,
            "revenue_volatility": 0.12,
            "cogs_ratio_mean": 0.50,
            "cogs_ratio_std": 0.04,
            "opex_ratio_mean": 0.28,
            "opex_ratio_std": 0.05,
            "asset_turnover_mean": 0.8,
            "asset_turnover_std": 0.1,
            "leverage_mean": 0.5,
            "leverage_std": 0.1,
            "dso_mean": 50.0,
            "dso_std": 5.0,
            "dio_mean": 40.0,
            "dio_std": 4.0,
            "dpo_mean": 45.0,
            "dpo_std": 4.0,
            "depreciation_rate_mean": 0.07,
            "depreciation_rate_std": 0.01
        }
    }
}

class SimulationConfig:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_data = DEFAULT_CONFIG.copy()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_path = os.path.join(current_dir, config_path)
        
        if os.path.exists(resolved_path):
            try:
                import yaml
                with open(resolved_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        self.config_data.update(loaded)
                        logger.info(f"Loaded configuration from {resolved_path}")
            except ImportError:
                logger.warning("PyYAML not installed. Using default fallback configuration.")
            except Exception as e:
                logger.error(f"Error loading {resolved_path}: {e}. Using defaults.")
        else:
            logger.info("Configuration file not found. Using defaults.")
            
        self.seed = self.config_data["seed"]
        self.num_normal = self.config_data["num_normal"]
        self.num_fraud = self.config_data["num_fraud"]
        self.start_year = self.config_data["start_year"]
        self.end_year = self.config_data["end_year"]
        self.pareto_alpha = self.config_data["pareto_alpha"]
        self.sectors = self.config_data["sectors"]
        
        self.set_seed()
        
    def set_seed(self) -> None:
        logger.info(f"Setting random seed to {self.seed} for reproducibility")
        random.seed(self.seed)
        np.random.seed(self.seed)

config = SimulationConfig()
