"""
Position Sizing Module for Calibrated Signal System

Implements Kelly Criterion position sizing using calibrated probabilities.
Includes risk management with position limits and confidence-based scaling.

Author: Signal System
Date: January 29, 2026
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculates optimal position sizes using Kelly Criterion with calibrated probabilities.
    
    Features:
    - Kelly Criterion with fractional sizing (quarter-kelly default)
    - Confidence-based scaling (higher confidence = larger positions)
    - Position limits (max positions, max capital deployed)
    - Risk management rules
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,  # Quarter-Kelly (25% of full Kelly)
        min_position_size: float = 0.01,  # 1% minimum
        max_position_size: float = 0.10,  # 10% maximum per position
        max_positions: int = 10,  # Max concurrent positions
        max_capital_deployed: float = 0.50,  # Max 50% of capital in use
        win_return: float = 0.02,  # Expected +2% win
        loss_return: float = -0.01,  # Expected -1% loss
    ):
        """
        Initialize position sizer with risk parameters.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter-Kelly)
            min_position_size: Minimum position size as fraction of capital
            max_position_size: Maximum position size as fraction of capital
            max_positions: Maximum number of concurrent positions
            max_capital_deployed: Maximum total capital deployed at once
            win_return: Expected return on winning trades (e.g., 0.02 = +2%)
            loss_return: Expected return on losing trades (e.g., -0.01 = -1%)
        """
        self.kelly_fraction = kelly_fraction
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.max_capital_deployed = max_capital_deployed
        self.win_return = win_return
        self.loss_return = loss_return
        
        logger.info(f"PositionSizer initialized:")
        logger.info(f"  Kelly Fraction: {kelly_fraction:.1%} (fractional Kelly)")
        logger.info(f"  Position Size Range: {min_position_size:.1%} - {max_position_size:.1%}")
        logger.info(f"  Max Positions: {max_positions}")
        logger.info(f"  Max Capital Deployed: {max_capital_deployed:.1%}")
        logger.info(f"  Expected Win/Loss: +{win_return:.1%} / {loss_return:.1%}")
    
    def calculate_kelly_fraction(self, win_prob: float, win_return: Optional[float] = None, 
                                 loss_return: Optional[float] = None) -> float:
        """
        Calculate Kelly Criterion fraction for a given win probability.
        
        Kelly Formula: f = (p*b - q) / b
        Where:
            f = fraction of capital to bet
            p = probability of winning
            q = probability of losing (1-p)
            b = ratio of win/loss magnitude (odds)
        
        Args:
            win_prob: Probability of winning (calibrated)
            win_return: Expected return on win (default: self.win_return)
            loss_return: Expected return on loss (default: self.loss_return)
        
        Returns:
            Kelly fraction (full Kelly, before applying fractional sizing)
        """
        if win_return is None:
            win_return = self.win_return
        if loss_return is None:
            loss_return = self.loss_return
        
        # Kelly Criterion calculation
        p = win_prob
        q = 1 - win_prob
        b = abs(win_return / loss_return)  # Win/loss ratio (e.g., 2% / 1% = 2.0)
        
        kelly = (p * b - q) / b
        
        # Kelly can be negative (unfavorable bet) - return 0 in that case
        return max(0, kelly)
    
    def calculate_position_size(self, win_prob: float, confidence_scaling: bool = True) -> float:
        """
        Calculate position size for a signal using Kelly Criterion.
        
        Args:
            win_prob: Calibrated win probability (e.g., 0.35 = 35%)
            confidence_scaling: If True, scale position by confidence level
        
        Returns:
            Position size as fraction of capital (e.g., 0.05 = 5%)
        """
        # Calculate full Kelly
        full_kelly = self.calculate_kelly_fraction(win_prob)
        
        # Apply fractional Kelly (e.g., quarter-Kelly)
        fractional_kelly = full_kelly * self.kelly_fraction
        
        # Apply confidence-based scaling if enabled
        if confidence_scaling:
            # Scale based on confidence level
            # Higher confidence (45-51%) → full size
            # Lower confidence (32-40%) → reduced size
            confidence_min = 0.32  # Threshold minimum
            confidence_max = 0.51  # Maximum observed
            
            # Linear scaling: 32% conf → 0.7x size, 51% conf → 1.0x size
            scale = 0.7 + 0.3 * ((win_prob - confidence_min) / (confidence_max - confidence_min))
            scale = np.clip(scale, 0.7, 1.0)
            
            fractional_kelly *= scale
        
        # Apply min/max limits
        position_size = np.clip(fractional_kelly, self.min_position_size, self.max_position_size)
        
        return position_size
    
    def add_position_sizes(self, signals_df: pl.DataFrame, confidence_col: str = "pred_proba") -> pl.DataFrame:
        """
        Add position sizing columns to signals dataframe.
        
        Args:
            signals_df: Polars DataFrame with signals
            confidence_col: Column name containing calibrated probabilities
        
        Returns:
            DataFrame with added columns:
                - kelly_fraction: Full Kelly Criterion fraction
                - position_size_pct: Recommended position size (%)
                - position_size_scaled: Position size with confidence scaling
                - kelly_category: Risk category (Conservative/Moderate/Aggressive)
        """
        logger.info(f"Calculating position sizes for {len(signals_df)} signals...")
        
        # Calculate full Kelly for each signal
        kelly_fractions = [
            self.calculate_kelly_fraction(prob) 
            for prob in signals_df[confidence_col].to_list()
        ]
        
        # Calculate fractional Kelly (without confidence scaling)
        fractional_kelly = [k * self.kelly_fraction for k in kelly_fractions]
        fractional_kelly = [
            np.clip(k, self.min_position_size, self.max_position_size) 
            for k in fractional_kelly
        ]
        
        # Calculate position size with confidence scaling
        position_sizes = [
            self.calculate_position_size(prob, confidence_scaling=True)
            for prob in signals_df[confidence_col].to_list()
        ]
        
        # Categorize by Kelly fraction
        categories = []
        for k in kelly_fractions:
            if k < 0.1:
                categories.append("Conservative")
            elif k < 0.3:
                categories.append("Moderate")
            else:
                categories.append("Aggressive")
        
        # Add columns to dataframe
        result_df = signals_df.with_columns([
            pl.Series("kelly_fraction", kelly_fractions),
            pl.Series("position_size_pct", fractional_kelly),
            pl.Series("position_size_scaled", position_sizes),
            pl.Series("kelly_category", categories),
        ])
        
        # Calculate expected value per signal
        expected_values = [
            prob * self.win_return + (1 - prob) * self.loss_return
            for prob in signals_df[confidence_col].to_list()
        ]
        result_df = result_df.with_columns([
            pl.Series("expected_value", expected_values)
        ])
        
        logger.info(f"✅ Position sizing complete")
        logger.info(f"   Avg Position Size: {np.mean(position_sizes):.2%}")
        logger.info(f"   Position Size Range: {np.min(position_sizes):.2%} - {np.max(position_sizes):.2%}")
        logger.info(f"   Avg Expected Value: {np.mean(expected_values):.2%}")
        
        return result_df
    
    def apply_portfolio_limits(self, signals_df: pl.DataFrame, sort_by: str = "expected_value") -> pl.DataFrame:
        """
        Apply portfolio-level position limits.
        
        - Select top N signals (by expected value or confidence)
        - Ensure total capital deployed <= max_capital_deployed
        - Mark selected signals for execution
        
        Args:
            signals_df: DataFrame with position_size_scaled column
            sort_by: Column to sort by for selection ("expected_value" or "pred_proba")
        
        Returns:
            DataFrame with added columns:
                - selected: Boolean indicating if signal should be traded
                - adjusted_position_size: Position size after portfolio limits applied
        """
        logger.info(f"Applying portfolio limits...")
        logger.info(f"  Max Positions: {self.max_positions}")
        logger.info(f"  Max Capital: {self.max_capital_deployed:.1%}")
        
        # Sort by selection criteria (descending)
        sorted_df = signals_df.sort(sort_by, descending=True)
        
        # Initialize selection
        selected = []
        adjusted_sizes = []
        cumulative_capital = 0.0
        
        for i, row in enumerate(sorted_df.iter_rows(named=True)):
            position_size = row["position_size_scaled"]
            
            # Check if we can add this position
            can_add = (
                len([s for s in selected if s]) < self.max_positions and
                cumulative_capital + position_size <= self.max_capital_deployed
            )
            
            if can_add:
                selected.append(True)
                adjusted_sizes.append(position_size)
                cumulative_capital += position_size
            else:
                selected.append(False)
                adjusted_sizes.append(0.0)
        
        # Add columns in original order (need to unsort)
        sorted_df = sorted_df.with_columns([
            pl.Series("selected", selected),
            pl.Series("adjusted_position_size", adjusted_sizes),
        ])
        
        # Restore original order if needed (assuming pair_address maintains order)
        if "pair_address" in sorted_df.columns:
            # Create mapping to restore order
            original_order = signals_df["pair_address"].to_list()
            sorted_df = sorted_df.sort(
                by="pair_address",
                maintain_order=True,
            )
        
        n_selected = sum(selected)
        total_capital = cumulative_capital
        
        logger.info(f"✅ Portfolio limits applied")
        logger.info(f"   Selected: {n_selected}/{len(signals_df)} signals")
        logger.info(f"   Total Capital Deployed: {total_capital:.1%}")
        
        return sorted_df
    
    def generate_position_report(self, signals_df: pl.DataFrame, output_path: Optional[str] = None) -> str:
        """
        Generate a detailed position sizing report.
        
        Args:
            signals_df: DataFrame with position sizing columns
            output_path: Optional path to save report (markdown)
        
        Returns:
            Report text (markdown formatted)
        """
        report_lines = [
            "# Position Sizing Report",
            f"**Generated:** {Path.cwd()}",
            f"**Signals Analyzed:** {len(signals_df)}",
            "",
            "## Configuration",
            f"- Kelly Fraction: {self.kelly_fraction:.1%} (fractional Kelly)",
            f"- Position Range: {self.min_position_size:.1%} - {self.max_position_size:.1%}",
            f"- Max Positions: {self.max_positions}",
            f"- Max Capital: {self.max_capital_deployed:.1%}",
            f"- Expected Win/Loss: +{self.win_return:.1%} / {self.loss_return:.1%}",
            "",
        ]
        
        if "selected" in signals_df.columns:
            selected_df = signals_df.filter(pl.col("selected") == True)
            report_lines.extend([
                "## Portfolio Selection",
                f"- **Selected Signals:** {len(selected_df)}/{len(signals_df)}",
                f"- **Total Capital Deployed:** {selected_df['adjusted_position_size'].sum():.1%}",
                f"- **Avg Position Size:** {selected_df['adjusted_position_size'].mean():.2%}",
                f"- **Total Expected Value:** {selected_df['expected_value'].sum():.2%}",
                "",
            ])
        
        # Summary statistics
        report_lines.extend([
            "## Position Sizing Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Avg Kelly Fraction | {signals_df['kelly_fraction'].mean():.2%} |",
            f"| Avg Position Size | {signals_df['position_size_pct'].mean():.2%} |",
            f"| Avg Scaled Position | {signals_df['position_size_scaled'].mean():.2%} |",
            f"| Avg Expected Value | {signals_df['expected_value'].mean():.3%} |",
            f"| Min Position Size | {signals_df['position_size_scaled'].min():.2%} |",
            f"| Max Position Size | {signals_df['position_size_scaled'].max():.2%} |",
            "",
        ])
        
        # Category breakdown
        if "kelly_category" in signals_df.columns:
            report_lines.extend([
                "## Risk Categories",
                "",
                "| Category | Count | Avg Position | Avg EV |",
                "|----------|-------|--------------|--------|",
            ])
            
            for category in ["Conservative", "Moderate", "Aggressive"]:
                cat_df = signals_df.filter(pl.col("kelly_category") == category)
                if len(cat_df) > 0:
                    report_lines.append(
                        f"| {category} | {len(cat_df)} | "
                        f"{cat_df['position_size_scaled'].mean():.2%} | "
                        f"{cat_df['expected_value'].mean():.3%} |"
                    )
            
            report_lines.append("")
        
        # Top signals
        if "selected" in signals_df.columns:
            selected_df = signals_df.filter(pl.col("selected") == True)
            if len(selected_df) > 0:
                top_signals = selected_df.sort("expected_value", descending=True).head(10)
                
                report_lines.extend([
                    "## Top 10 Selected Signals",
                    "",
                    "| Rank | Pair | Confidence | Position | EV |",
                    "|------|------|------------|----------|-----|",
                ])
                
                for i, row in enumerate(top_signals.iter_rows(named=True), 1):
                    pair = row.get("pair_address", "N/A")[:12] + "..."
                    conf = row.get("pred_proba", 0)
                    pos = row.get("adjusted_position_size", 0)
                    ev = row.get("expected_value", 0)
                    report_lines.append(
                        f"| {i} | {pair} | {conf:.1%} | {pos:.2%} | {ev:+.2%} |"
                    )
                
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            Path(output_path).write_text(report_text)
            logger.info(f"📄 Report saved to: {output_path}")
        
        return report_text


def add_position_sizing_to_signals(
    signal_file: str,
    output_file: Optional[str] = None,
    kelly_fraction: float = 0.25,
    max_positions: int = 10,
    max_capital: float = 0.50,
    apply_limits: bool = True,
    generate_report: bool = True,
) -> pl.DataFrame:
    """
    Main function to add position sizing to a signals CSV file.
    
    Args:
        signal_file: Path to signals CSV file
        output_file: Optional output path (defaults to signal_file with _positioned suffix)
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter-Kelly)
        max_positions: Maximum number of concurrent positions
        max_capital: Maximum total capital deployed
        apply_limits: Whether to apply portfolio limits and select signals
        generate_report: Whether to generate position sizing report
    
    Returns:
        DataFrame with position sizing columns added
    """
    logger.info(f"="*80)
    logger.info(f"POSITION SIZING: {signal_file}")
    logger.info(f"="*80)
    
    # Load signals
    signals_df = pl.read_csv(signal_file)
    logger.info(f"Loaded {len(signals_df)} signals")
    
    # Check for required column
    if "pred_proba" not in signals_df.columns:
        raise ValueError("Signals file must contain 'pred_proba' column with calibrated probabilities")
    
    # Initialize position sizer
    sizer = PositionSizer(
        kelly_fraction=kelly_fraction,
        max_positions=max_positions,
        max_capital_deployed=max_capital,
    )
    
    # Add position sizes
    result_df = sizer.add_position_sizes(signals_df)
    
    # Apply portfolio limits if requested
    if apply_limits:
        result_df = sizer.apply_portfolio_limits(result_df)
    
    # Generate report if requested
    if generate_report:
        report_path = signal_file.replace(".csv", "_position_report.md")
        sizer.generate_position_report(result_df, output_path=report_path)
    
    # Save output
    if output_file is None:
        output_file = signal_file.replace(".csv", "_positioned.csv")
    
    result_df.write_csv(output_file)
    logger.info(f"💾 Saved positioned signals to: {output_file}")
    
    return result_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python position_sizing.py <signal_file.csv> [options]")
        print("\nOptions:")
        print("  --kelly-fraction 0.25    Fraction of Kelly to use (default: 0.25)")
        print("  --max-positions 10       Max concurrent positions (default: 10)")
        print("  --max-capital 0.50       Max capital deployed (default: 0.50)")
        print("  --no-limits              Don't apply portfolio limits")
        print("  --no-report              Don't generate report")
        print("\nExample:")
        print("  python position_sizing.py signals_2026-01-29_04-39-38.csv")
        print("  python position_sizing.py signals_live.csv --max-positions 15 --max-capital 0.60")
        sys.exit(1)
    
    signal_file = sys.argv[1]
    
    # Parse optional arguments
    kelly_fraction = 0.25
    max_positions = 10
    max_capital = 0.50
    apply_limits = True
    generate_report = True
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--kelly-fraction":
            kelly_fraction = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--max-positions":
            max_positions = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--max-capital":
            max_capital = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--no-limits":
            apply_limits = False
            i += 1
        elif sys.argv[i] == "--no-report":
            generate_report = False
            i += 1
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)
    
    # Process signals
    result_df = add_position_sizing_to_signals(
        signal_file,
        kelly_fraction=kelly_fraction,
        max_positions=max_positions,
        max_capital=max_capital,
        apply_limits=apply_limits,
        generate_report=generate_report,
    )
    
    print("\n" + "="*80)
    print("✅ POSITION SIZING COMPLETE")
    print("="*80)
