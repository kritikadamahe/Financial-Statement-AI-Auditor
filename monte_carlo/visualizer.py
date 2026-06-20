import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import List

# Configure logger
logger = logging.getLogger("MonteCarloVisualizer")

class DatasetVisualizer:
    """
    Generates publication-quality charts (300 DPI) for the synthetic
    financial statement population containing 6 fraud typologies.
    """
    def __init__(self, panel_df: pd.DataFrame, company_df: pd.DataFrame, output_dir: str = "monte_carlo"):
        self.panel_df = panel_df.copy()
        self.company_df = company_df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set styling defaults
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 14,
            'figure.dpi': 300
        })

    def plot_fraud_distribution(self) -> str:
        """Donut chart showing breakdown of company labels (Normal + 6 Fraud types)."""
        plt.figure(figsize=(7, 6))
        
        counts = self.company_df["fraud_type"].value_counts()
        labels = [c.replace('_', ' ').title() for c in counts.index]
        
        # Expanded color palette to cover Normal + 6 Fraud typologies
        colors = ['#2b5c8f', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d']
        colors = colors[:len(counts)]
        
        plt.pie(
            counts, 
            labels=labels, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=colors,
            pctdistance=0.85,
            wedgeprops=dict(width=0.4, edgecolor='w', linewidth=1.5)
        )
        
        plt.title("Financial Statement Label Distribution\n(Normal vs. 6 Injected Fraud Typologies)", fontweight='bold', pad=15)
        
        out_path = os.path.join(self.output_dir, "fraud_distribution.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved fraud distribution chart to {out_path}")
        return out_path

    def plot_metric_distributions(self) -> str:
        """Histograms of Revenue and Operating Cash Flow for normal vs. fraud."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        snap_df = self.panel_df[self.panel_df["year"] == 2024]
        
        # Revenue distribution (log scale because Pareto distribution spans orders of magnitude)
        sns.histplot(
            data=snap_df, 
            x="revenue", 
            hue="is_fraud", 
            kde=True, 
            bins=30, 
            palette={0: '#2b5c8f', 1: '#d95f02'}, 
            alpha=0.6, 
            ax=axes[0]
        )
        axes[0].set_title("Revenue Distribution (FY 2024 - Pareto Scale)", fontweight='bold')
        axes[0].set_xlabel("Revenue (Log-Scale USD)")
        axes[0].set_xscale('log')
        axes[0].get_legend().set_title("Label (0=Normal, 1=Fraud)")
        
        # OCF distribution
        sns.histplot(
            data=snap_df, 
            x="operating_cash_flow", 
            hue="is_fraud", 
            kde=True, 
            bins=30, 
            palette={0: '#2b5c8f', 1: '#d95f02'}, 
            alpha=0.6, 
            ax=axes[1]
        )
        axes[1].set_title("Operating Cash Flow Distribution (FY 2024)", fontweight='bold')
        axes[1].set_xlabel("Operating Cash Flow (USD)")
        axes[1].get_legend().set_title("Label (0=Normal, 1=Fraud)")
        
        plt.suptitle("Financial Metric Distributions (Normal vs. Fraudulent)", fontweight='bold', y=1.02)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "metric_distributions.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved metric distributions chart to {out_path}")
        return out_path

    def plot_correlation_heatmap(self) -> str:
        """Correlation heatmap of engineered ratios."""
        plt.figure(figsize=(10, 8))
        
        ratio_cols = [
            "_CurrentRatio_mean", "_DebtToEquity_mean", 
            "_NetMargin_mean", "_GrossMargin_mean", 
            "_RevOCF_corr", "_NI_OCF_corr", "_slope_divergence"
        ]
        
        presentation_cols = [
            "Current Ratio Mean", "Debt-to-Equity Mean",
            "Net Profit Margin Mean", "Gross Margin Mean",
            "Revenue-OCF Corr", "Net Income-OCF Corr", "Slope Divergence"
        ]
        
        corr_df = self.company_df[ratio_cols].copy()
        corr_df.columns = presentation_cols
        corr_matrix = corr_df.corr()
        
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            vmin=-1.0, 
            vmax=1.0, 
            linewidths=0.5,
            cbar_kws={'label': 'Pearson Correlation'}
        )
        
        plt.title("Pearson Correlation Heatmap of Engineered Financial Ratios", fontweight='bold', pad=15)
        
        out_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlation heatmap to {out_path}")
        return out_path

    def plot_growth_distributions(self) -> str:
        """Growth distribution density contrast for Revenue vs OCF."""
        plt.figure(figsize=(8, 6))
        
        sns.kdeplot(
            data=self.panel_df[self.panel_df["is_fraud"] == 0], 
            x="revenue_growth", 
            label="Normal Revenue Growth", 
            color="#2b5c8f", 
            fill=True, 
            alpha=0.3
        )
        sns.kdeplot(
            data=self.panel_df[self.panel_df["is_fraud"] == 0], 
            x="ocf_growth", 
            label="Normal OCF Growth", 
            color="#66a61e", 
            fill=True, 
            alpha=0.3
        )
        
        sns.kdeplot(
            data=self.panel_df[self.panel_df["fraud_type"] == "revenue_inflation"], 
            x="revenue_growth", 
            label="Inflation Revenue Growth (Fraud)", 
            color="#d95f02", 
            linestyle="--", 
            linewidth=2
        )
        
        plt.xlim(-0.5, 1.5)
        plt.title("Annual Revenue and OCF Growth Rate Density Curves", fontweight='bold', pad=15)
        plt.xlabel("Growth Rate (%)")
        plt.ylabel("Density")
        plt.legend(loc="upper right", frameon=True)
        
        out_path = os.path.join(self.output_dir, "growth_distributions.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved growth distributions density chart to {out_path}")
        return out_path

    def plot_fraud_type_comparisons(self) -> str:
        """Boxplots comparing key features across all 6 fraud types."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        plot_df = self.company_df.copy()
        plot_df["fraud_type_label"] = plot_df["fraud_type"].apply(
            lambda x: x.replace('_', ' ').title()
        )
        
        # Subplot 1: Slope Divergence
        sns.boxplot(
            data=plot_df, 
            x="fraud_type_label", 
            y="_slope_divergence", 
            hue="fraud_type_label",
            legend=False,
            palette="Set2", 
            ax=axes[0, 0]
        )
        axes[0, 0].set_title("Slope Divergence by Label", fontweight='bold')
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("Slope Divergence")
        axes[0, 0].tick_params(axis='x', rotation=30, labelsize=9)
        
        # Subplot 2: Revenue Volatility
        sns.boxplot(
            data=plot_df, 
            x="fraud_type_label", 
            y="Revenue_yoy_volatility", 
            hue="fraud_type_label",
            legend=False,
            palette="Set2", 
            ax=axes[0, 1]
        )
        axes[0, 1].set_title("Revenue YoY Volatility", fontweight='bold')
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("Volatility (%)")
        axes[0, 1].tick_params(axis='x', rotation=30, labelsize=9)
        
        # Subplot 3: Debt-to-Equity Mean
        sns.boxplot(
            data=plot_df, 
            x="fraud_type_label", 
            y="_DebtToEquity_mean", 
            hue="fraud_type_label",
            legend=False,
            palette="Set2", 
            ax=axes[1, 0]
        )
        axes[1, 0].set_title("Debt-to-Equity Mean", fontweight='bold')
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_ylabel("Ratio")
        axes[1, 0].tick_params(axis='x', rotation=30, labelsize=9)
        
        # Subplot 4: Net Income vs OCF correlation
        sns.boxplot(
            data=plot_df, 
            x="fraud_type_label", 
            y="_NI_OCF_corr", 
            hue="fraud_type_label",
            legend=False,
            palette="Set2", 
            ax=axes[1, 1]
        )
        axes[1, 1].set_title("Net Income vs OCF Correlation", fontweight='bold')
        axes[1, 1].set_xlabel("")
        axes[1, 1].set_ylabel("Pearson r")
        axes[1, 1].tick_params(axis='x', rotation=30, labelsize=9)
        
        plt.suptitle("Engineered Financial Signatures across 6 Fraud Typologies", fontweight='bold', y=1.01)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "fraud_type_comparisons.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved fraud type comparisons chart to {out_path}")
        return out_path

    def generate_all_plots(self) -> List[str]:
        """Runs the entire visualization generation suite."""
        logger.info("Starting publication visualization generation suite...")
        paths = [
            self.plot_fraud_distribution(),
            self.plot_metric_distributions(),
            self.plot_correlation_heatmap(),
            self.plot_growth_distributions(),
            self.plot_fraud_type_comparisons()
        ]
        logger.info("Visualization suite completed successfully.")
        return paths
