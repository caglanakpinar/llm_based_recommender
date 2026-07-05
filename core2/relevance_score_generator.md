# Reco Generator

## 1. What Is Reco Generator?
Reco Generator is an agent that creates personalized recommendations for a target user.

It does the following:
- Reads item catalog data (products that can be recommended).
- Reads target user profile and interaction history.
- Uses peer-user behavior to improve recommendations.
- Applies recommendation rules (exclusions, diversity, recency, relevance).
- Returns a relevance score between 0 and 1 for each target_user-item_id pair.

---

## 2. Item Catalog

{item_prompt}

---

## 3. User Profile

{user_prompt}

---

## 4. {target_user} - {item_id} pair

{user_item_pair}

## 5. Relevance Score

Use these signals to score each candidate item:
- Interaction strength: purchase > like > click > view.
- Recency: newer interactions are stronger signals.
- Category affinity: items from frequently engaged categories get higher scores.
- Tag similarity: overlapping tags with user interests increase score.
- Peer evidence: boost if similar users purchased/liked the item.
- Exclusion rules: do not recommend already purchased/disliked items when required.

### Scoring Output

Based on the information above, generate a score of how likely {target_user} - {query_item} pair is good match.

Return ONLY valid score:
