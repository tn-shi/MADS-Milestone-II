"""Shared configuration for the recession forecasting pipeline.

Contains FRED series definitions, country codes, and series ID construction
functions used across Dataset, Preprocessing, and Baseline notebooks.
"""

from dataclasses import dataclass, field
from typing import Literal

# Type alias for supported feature engineering operations
FeatureOp = Literal["log_diff", "first_diff", "amplitude_deviation", "rolling_stats"]


@dataclass(frozen=True)
class SeriesConfig:
    """Configuration for a FRED series and its feature engineering.

    Attributes:
        prefix: Characters before the country code in series ID (default: "")
        suffix: Characters after the country code in series ID (default: "")
        use_iso3: Use 3-letter (True) or 2-letter (False) country codes (default: True)
        is_global: If True, series is not country-specific, e.g., VIX (default: False)
        agg_method: Aggregation method for frequency conversion, e.g., "avg" (default: None)
        suffix_overrides: Country-specific suffix exceptions (default: {})
        iso_overrides: Country-specific ISO code exceptions (default: {})
        custom_id: Country-specific custom series IDs that bypass pattern construction (default: {})
        feature_ops: Feature operation(s) to apply - single string or list (default: [])
        lagged: If True, add lagged features for this indicator (default: False)
        publication_lag: Months to shift data forward to account for release delay (default: 0)
    """

    # FRED series construction
    prefix: str = ""
    suffix: str = ""
    use_iso3: bool = True
    is_global: bool = False
    agg_method: str | None = None
    suffix_overrides: dict[str, str] = field(default_factory=dict)
    iso_overrides: dict[str, str] = field(default_factory=dict)
    custom_id: dict[str, str] = field(default_factory=dict)

    # Feature engineering configuration
    feature_ops: FeatureOp | list[FeatureOp] = field(default_factory=list)
    lagged: bool = False
    publication_lag: int = 0

    def __post_init__(self):
        """Normalize feature_ops to a list."""
        if isinstance(self.feature_ops, str):
            object.__setattr__(self, "feature_ops", [self.feature_ops])

    def has_op(self, op: FeatureOp) -> bool:
        """Check if this series should have the given operation applied."""
        return op in self.feature_ops


G7_COUNTRY_CODES = {
    "USA": {"iso2": "US", "iso3": "USA"},
    "Canada": {"iso2": "CA", "iso3": "CAN"},
    "UK": {"iso2": "GB", "iso3": "GBR"},
    "France": {"iso2": "FR", "iso3": "FRA"},
    "Germany": {"iso2": "DE", "iso3": "DEU"},
    "Italy": {"iso2": "IT", "iso3": "ITA"},
    "Japan": {"iso2": "JP", "iso3": "JPN"},
    "South Korea": {"iso2": "KR", "iso3": "KOR"},
    "Australia": {"iso2": "AU", "iso3": "AUS"},
}

SERIES_CONFIG: dict[str, SeriesConfig] = {
    # Real GDP, quarterly, in national currency units, seasonally adj.
    # USA URL: https://fred.stlouisfed.org/series/NAEXKP01USQ652S
    # Publication lag: ~3 months (advance estimate released ~30 days after quarter end)
    "real_gdp": SeriesConfig(
        prefix="NAEXKP01",
        suffix="Q189S",
        use_iso3=False,
        suffix_overrides={"USA": "Q652S", "UK": "Q652S"},
        custom_id={"Australia": "AUSGDPRQDSMEI"},
        feature_ops="log_diff",
        publication_lag=3,
    ),
    # Consumer Price Index, 2015=100, monthly
    # USA URL: https://fred.stlouisfed.org/series/USACPIALLMINMEI
    "cpi": SeriesConfig(
        suffix="CPIALLMINMEI",
        feature_ops="log_diff",
        custom_id={"Australia": "AUSCPALTT01IXOBSAQ"},  # Quartely, will ffill
    ),
    # Unemployment Rate (%), Monthly, Seasonally Adj.
    # USA URL: https://fred.stlouisfed.org/series/LRHUTTTTUSM156S
    # Lagged: Labor market conditions lag economic turning points
    "unemployment_rate": SeriesConfig(
        prefix="LRHUTTTT",
        suffix="M156S",
        use_iso3=False,
        feature_ops="first_diff",
        lagged=True,
    ),
    # Economic Policy Uncertainty Index, Monthly
    # USA URL: https://fred.stlouisfed.org/series/USEPUINDXM
    # Japan's FRED data is discontinued (2016); Australia is not on FRED.
    # Both loaded from local files. Source: https://policyuncertainty.com/
    # Lagged: Policy uncertainty signals precede market stress
    "epu": SeriesConfig(
        suffix="EPUINDXM",
        use_iso3=False,
        suffix_overrides={"France": "EUINDXM"},
        iso_overrides={
            "Japan": "JPN",
            "UK": "UK",
            "Canada": "CAN",
            "South Korea": "KOREA",
        },
        feature_ops=["log_diff", "rolling_stats"],
        lagged=True,
    ),
    # 10 Year Government Bond Interest Rates, Monthly
    # USA URL: https://fred.stlouisfed.org/series/IRLTLT01USM156N
    # Lagged: Long-term rates reflect financial conditions
    "10_yr_yld": SeriesConfig(
        prefix="IRLTLT01",
        suffix="M156N",
        use_iso3=False,
        feature_ops="first_diff",
        lagged=True,
    ),
    # 3-Month Interbank Interest Rate, Monthly
    # USA URL: https://fred.stlouisfed.org/series/IR3TIB01USM156N
    # This is the interest rate that banks charge other banks for a 90-day loan
    # not using 3 month government bond yield as it's not available for all G7 countries
    # Lagged: Short-term rates reflect monetary policy stance
    "3_mo_yld": SeriesConfig(
        prefix="IR3TIB01",
        suffix="M156N",
        use_iso3=False,
        feature_ops="first_diff",
        custom_id={"Japan": "IR3TCD01JPM156N"},  # 3 month rate for CDs
        lagged=True,
    ),
    # OECD based Recession Indicators, Monthly
    # USA URL: https://fred.stlouisfed.org/series/USAREC
    # Discontinued since 2022 but can be used for historical analysis
    # Can find alternatives if needed
    "oecd_rec": SeriesConfig(
        suffix="REC",
    ),
    # Industrial Activity Index, Monthly, Seasonally Adj.
    # USA URL: https://fred.stlouisfed.org/series/USAPRINTO01IXOBM
    # Lagged: Industrial output tracks real economy momentum
    "ind_out": SeriesConfig(
        suffix="PROINDMISMEI",
        feature_ops="log_diff",
        lagged=True,
        custom_id={"Australia": "AUSPROINDQISMEI"},  # Quarterly
    ),
    # Composite Consumer Confidence Amplitude, Monthly, Seasonally Adj.
    # USA URL: https://fred.stlouisfed.org/series/CSCICP03USM665S
    # Normal is 100 (amplitude-adjusted index)
    # Lagged: Consumer sentiment is a leading indicator
    "comp_consumer_conf": SeriesConfig(
        prefix="CSCICP03",
        suffix="M665S",
        use_iso3=False,
        feature_ops="amplitude_deviation",
        lagged=True,
    ),
    # Car Registration for Passenger Cars, Monthly, Seasonally Adj.
    # USA URL: https://fred.stlouisfed.org/series/USASLRTCR03GPSAM
    "pcar_reg": SeriesConfig(
        suffix="SLRTCR03GPSAM",
    ),
    # VIX - Daily data, aggregated to monthly via average
    # URL: https://fred.stlouisfed.org/series/VIXCLS
    # Lagged: Market volatility signals financial stress
    "vix": SeriesConfig(
        prefix="VIXCLS",
        use_iso3=False,
        is_global=True,
        agg_method="avg",
        feature_ops=["log_diff", "rolling_stats"],
        lagged=True,
    ),
    # Spot Crude Oil Price: West Texas Intermediate (WTI), Monthly
    # URL: https://fred.stlouisfed.org/series/WTISPLC
    "oil": SeriesConfig(
        prefix="WTISPLC",
        use_iso3=False,
        is_global=True,
        feature_ops="log_diff",
    ),
    # Producer Price Index by Commodity: Special Indexes: Copper and Copper Products
    # Index: 1982=100, Monthly
    # URL: https://fred.stlouisfed.org/series/WPUSI019011
    "copper": SeriesConfig(
        prefix="WPUSI019011",
        use_iso3=False,
        is_global=True,
        feature_ops="log_diff",
        lagged=True,
    ),
    # Proxy for Gold, Gold Spot Price is no longer available due to licensing from
    # the London Bullion Market Association
    # Producer Price Index for Jewelry (Gold and Platinum) and Silverware, Monthly
    # https://fred.stlouisfed.org/series/WPU159402
    "gps": SeriesConfig(
        prefix="WPU159402",
        use_iso3=False,
        is_global=True,
        feature_ops="log_diff",
    ),
    # OECD: Leading Indicators: Composite Leading Indicator: Amplitude Adjusted, Monthly
    # URL: https://fred.stlouisfed.org/series/USALOLITOAASTSAM
    # Lagged: Composite leading indicator designed to predict cycles
    "cli": SeriesConfig(
        suffix="LOLITOAASTSAM",
        feature_ops="amplitude_deviation",
        lagged=True,
    ),
    # Sales: Retail Trade: Total Retail Trade: Volume, Growth Rate Previous Peroid, Monthly
    # URL: https://fred.stlouisfed.org/series/USASLRTTO01GPSAM
    "retail_vol": SeriesConfig(
        suffix="SLRTTO01GPSAM",
        feature_ops="log_diff",
        lagged=True,
        custom_id={"Australia": "SLRTTO01AUQ189S"},  # Quarterly
    ),
    # Financial Market: Share Prices, Index 2015=100, Monthly
    # URL: https://fred.stlouisfed.org/series/SPASTT01USM661N
    "national_share_price": SeriesConfig(
        prefix="SPASTT01",
        suffix="M661N",
        use_iso3=False,
        feature_ops=["log_diff", "rolling_stats"],
        lagged=True,
    ),
}

# Configuration for loading local EPU data files
# Assumes Excel files have columns: year (1st), month (2nd), value (3rd)
LOCAL_EPU_CONFIG = {
    "Japan": "data/Japan_Policy_Uncertainty_Data.xlsx",
    "Australia": "data/Australia_Policy_Uncertainty_Data.xlsx",
}


def build_series_id(indicator: str, country: str) -> str:
    """Constructs a FRED series ID from template and country code.

    Args:
        indicator: The indicator type (e.g., 'real_gdp', 'cpi').
        country: The country name (e.g., 'USA', 'UK').

    Returns:
        The constructed FRED series ID string.

    Raises:
        ValueError: If indicator or country is not recognized.
    """
    if indicator not in SERIES_CONFIG:
        raise ValueError(f"Unknown indicator: {indicator}")
    if country not in G7_COUNTRY_CODES:
        raise ValueError(f"Unknown country: {country}")

    config = SERIES_CONFIG[indicator]
    codes = G7_COUNTRY_CODES[country]

    # Check for custom series ID first (bypasses pattern construction)
    if country in config.custom_id:
        return config.custom_id[country]

    # Global indicators don't use country codes
    if config.is_global:
        return f"{config.prefix}{config.suffix}"

    # Check for country-specific ISO override first
    if country in config.iso_overrides:
        code = config.iso_overrides[country]
    else:
        code = codes["iso3"] if config.use_iso3 else codes["iso2"]

    suffix = config.suffix_overrides.get(country, config.suffix)

    return f"{config.prefix}{code}{suffix}"


def build_series_dict(
    indicators: list[str] | None = None,
    countries: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Builds a nested dictionary of FRED series IDs.

    Args:
        indicators: List of indicators to include all by default
        countries: List of countries to include all by default

    Returns:
        Nested dict: {indicator: {country: series_id, country2: series_id2}, indicator2: ...}
        For global indicators, all countries map to the same series ID.
    """
    indicators = indicators or list(SERIES_CONFIG.keys())
    countries = countries or list(G7_COUNTRY_CODES.keys())

    series_ids_by_indicator = {}

    for indicator in indicators:
        config = SERIES_CONFIG[indicator]

        if config.is_global:
            # Global indicators use the same series ID for all countries
            global_series_id = build_series_id(indicator, countries[0])
            series_ids_by_country = {
                country: global_series_id for country in countries
            }
        else:
            series_ids_by_country = {
                country: build_series_id(indicator, country) for country in countries
            }

        series_ids_by_indicator[indicator] = series_ids_by_country

    return series_ids_by_indicator
