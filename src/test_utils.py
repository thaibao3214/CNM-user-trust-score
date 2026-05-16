from utils import calculate_trust_score
from utils import get_risk_level

fraud_prob = 0.18

trust_score = calculate_trust_score(fraud_prob)

risk = get_risk_level(trust_score)

print("Trust Score:", trust_score)
print("Risk Level:", risk)