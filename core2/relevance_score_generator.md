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
Each item contains these fields:
- item_id: unique identifier of the item.
- title: item name shown to users.
- category: item group (books, electronics, travel, etc.).
- tags: keywords that describe item attributes.
- price: numeric item price.
- description: short text describing the item.
- generated_prompt:
- generated_prompt_emb_vectors:

Example item prompt:

# Item Generated Prompt

Start with Item ID and describe the item using placeholders.

- Item ID is item_0101.
- Title of item_0101 is Product item_0101.
- Category of Product item_0101 is books.
- Tags of Product item_0101 are books, sample, tag1.
- Price of Product item_0101 is 49.99.
- Description of Product item_0101: A books product for testing recommendations.

## Single-Line Prompt Template
item_id: item_0101; title: Product item_0101; category: books; tags: books, sample, tag1; price: 49.99; description: A books product for testing recommendations.


---

## 3. User Profile
Each user contains these fields:
- user_id: unique user identifier.
- segment: user cohort (new_customer, returning_customer, vip, churn_risk, etc.).
- notes: free-text context about user interests/behavior.

Optional behavior context used by Reco Generator:
- target_user_interactions: the target user's actions (view, click, purchase, like, etc.).
- other_users_interactions: peer users and their actions for collaborative filtering.

Example user profile shape:

```json
{
  "user_id": "user_00001",
  "segment": "new_customer",
  "notes": "Sample notes for user_00001 in segment new_customer."
}
```

---

## 4. Relevance Score
Target user interactions template:

- {target_user} has interactions with items below:
  {interacted_items}

Use these signals to score each candidate item:
- Interaction strength: purchase > like > click > view.
- Recency: newer interactions are stronger signals.
- Category affinity: items from frequently engaged categories get higher scores.
- Tag similarity: overlapping tags with user interests increase score.
- Peer evidence: boost if similar users purchased/liked the item.
- Exclusion rules: do not recommend already purchased/disliked items when required.

### Scoring Output
Based on the information above, generate a relevance score for {target_user} - {query_item} pair.

Return ONLY valid score:
