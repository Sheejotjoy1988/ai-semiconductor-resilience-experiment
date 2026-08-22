import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 20000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "semiconductor_supply_chain.csv"
)

rng = np.random.default_rng(RANDOM_STATE)

countries = [
    "United States",
    "Taiwan",
    "South Korea",
    "Japan",
    "Netherlands",
    "China",
    "Vietnam",
    "Malaysia",
    "India",
    "Mexico"
]

products = [
    "Logic",
    "Memory",
    "Analog",
    "Power",
    "Microcontroller",
    "Sensors"
]

country_risk = {
    "United States": 0.25,
    "Taiwan": 0.55,
    "South Korea": 0.40,
    "Japan": 0.30,
    "Netherlands": 0.25,
    "China": 0.70,
    "Vietnam": 0.45,
    "Malaysia": 0.40,
    "India": 0.45,
    "Mexico": 0.40
}

records = []

for i in range(N_SAMPLES):

    country = rng.choice(countries)
    product = rng.choice(products)

    supplier_concentration = rng.beta(2.5, 3.0)

    lead_time_days = np.clip(
        rng.normal(65 + supplier_concentration * 35, 18),
        15,
        180
    )

    geopolitical_risk = np.clip(
        country_risk[country] + rng.normal(0, 0.12),
        0,
        1
    )

    logistics_risk = np.clip(
        rng.beta(2, 4) + geopolitical_risk * 0.20,
        0,
        1
    )

    demand_volatility = np.clip(
        rng.beta(2.5, 3.5),
        0,
        1
    )

    inventory_days = np.clip(
        rng.normal(50, 18),
        5,
        140
    )

    single_source_dependency = rng.binomial(
        1,
        np.clip(
            0.15 + supplier_concentration * 0.65,
            0,
            0.90
        )
    )

    capacity_utilization = np.clip(
        rng.normal(0.78, 0.12),
        0.35,
        1.0
    )

    supplier_quality_score = np.clip(
        rng.normal(0.85, 0.08),
        0.45,
        1.0
    )

    cyber_risk = np.clip(
        rng.beta(2, 5) + geopolitical_risk * 0.15,
        0,
        1
    )

    natural_disaster_risk = np.clip(
        rng.beta(1.8, 5),
        0,
        1
    )

    transport_delay_rate = np.clip(
        logistics_risk * 0.55 + rng.normal(0.10, 0.08),
        0,
        1
    )

    import_dependency = np.clip(
        rng.beta(3, 2),
        0,
        1
    )

    demand_supply_gap = np.clip(
        demand_volatility * 0.55
        + capacity_utilization * 0.25
        + rng.normal(0, 0.10),
        0,
        1
    )

    resilience_investment = np.clip(
        rng.beta(2.5, 3),
        0,
        1
    )

    digital_visibility = np.clip(
        rng.beta(3, 2.5),
        0,
        1
    )

    # Synthetic latent disruption mechanism.
    # Positive terms increase disruption probability.
    # Inventory, resilience investment, digital visibility,
    # and supplier quality reduce disruption probability.
    logit = (
        -4.0
        + 2.2 * supplier_concentration
        + 1.7 * geopolitical_risk
        + 1.5 * logistics_risk
        + 1.3 * demand_volatility
        + 1.1 * single_source_dependency
        + 1.2 * capacity_utilization
        + 1.1 * cyber_risk
        + 0.9 * natural_disaster_risk
        + 1.5 * transport_delay_rate
        + 1.0 * import_dependency
        + 1.4 * demand_supply_gap
        + 0.010 * lead_time_days
        - 0.012 * inventory_days
        - 1.4 * resilience_investment
        - 1.1 * digital_visibility
        - 1.0 * supplier_quality_score
        + rng.normal(0, 0.55)
    )

    disruption_probability = 1 / (1 + np.exp(-logit))

    disrupted = rng.binomial(
        1,
        disruption_probability
    )

    records.append({
        "supplier_id": f"SUP-{i+1:05d}",
        "country": country,
        "product_type": product,
        "supplier_concentration": round(supplier_concentration, 4),
        "lead_time_days": round(lead_time_days, 2),
        "geopolitical_risk": round(geopolitical_risk, 4),
        "logistics_risk": round(logistics_risk, 4),
        "demand_volatility": round(demand_volatility, 4),
        "inventory_days": round(inventory_days, 2),
        "single_source_dependency": single_source_dependency,
        "capacity_utilization": round(capacity_utilization, 4),
        "supplier_quality_score": round(supplier_quality_score, 4),
        "cyber_risk": round(cyber_risk, 4),
        "natural_disaster_risk": round(natural_disaster_risk, 4),
        "transport_delay_rate": round(transport_delay_rate, 4),
        "import_dependency": round(import_dependency, 4),
        "demand_supply_gap": round(demand_supply_gap, 4),
        "resilience_investment": round(resilience_investment, 4),
        "digital_visibility": round(digital_visibility, 4),
        "disrupted": disrupted
    })

df = pd.DataFrame(records)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 70)
print("SEMICONDUCTOR SUPPLY-CHAIN DATASET CREATED")
print("=" * 70)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Disruption rate: {df['disrupted'].mean():.2%}")

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst five rows:")
print(df.head())

print("\nClass distribution:")
print(df["disrupted"].value_counts())

print("\nDataset generation completed successfully.")
