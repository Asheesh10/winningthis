import pandas as pd
import numpy as np
import random
from pathlib import Path

# Goal: Generate data matching FEATURE_ORDER
# FEATURE_ORDER = [
#     "monthly_income",
#     "income_stability",
#     "rent_payment_consistency",
#     "utility_payment_rate",
#     "savings_ratio",
#     "transaction_volume",
#     "missed_payments",
#     "gpa",
# ]

def generate_compatible_data(n=1000):
    users = []
    for _ in range(n):
        # Base attributes
        monthly_income = round(random.uniform(5000, 50000), 2)
        income_stability = round(random.uniform(0.1, 1.0), 2)
        
        # Synthetic behavioral/financial stats
        rent_payment_consistency = round(random.uniform(0.5, 1.0), 2)
        utility_payment_rate = round(random.uniform(0.5, 1.0), 2)
        
        savings_balance = random.uniform(1000, 100000)
        savings_ratio = round(min(savings_balance / (monthly_income * 12 + 1), 1.0), 2)
        
        transaction_volume = random.randint(10, 500)
        missed_payments = random.choices([0, 1, 2, 3, 5], weights=[0.6, 0.2, 0.1, 0.05, 0.05])[0]
        gpa = round(random.uniform(2.0, 4.0), 2)

        # Logic for default (target)
        # Higher consistency/income -> Lower risk (0 in default)
        # Higher missed payments -> Higher risk (1 in default)
        risk_score = (
            (1 - income_stability) * 0.2 + 
            (1 - rent_payment_consistency) * 0.2 + 
            (missed_payments / 10) * 0.4 + 
            (1 - savings_ratio) * 0.2
        )
        # Add some noise
        risk_score += random.uniform(-0.1, 0.1)
        
        defaulted = 1 if risk_score > 0.4 else 0

        users.append({
            "monthly_income": monthly_income,
            "income_stability": income_stability,
            "rent_payment_consistency": rent_payment_consistency,
            "utility_payment_rate": utility_payment_rate,
            "savings_ratio": savings_ratio,
            "transaction_volume": transaction_volume,
            "missed_payments": missed_payments,
            "gpa": gpa,
            "defaulted": defaulted
        })

    return pd.DataFrame(users)

if __name__ == "__main__":
    df = generate_compatible_data(1200)
    # Save to current directory (will be app/ when run)
    output_path = Path("training_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows of compatible training data at {output_path}")
