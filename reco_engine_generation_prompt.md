# LLM-Based Recommendation Engine Generation Prompt

You are an expert recommendation system engineer. Your task is to **design and generate a personalized recommendation engine** that takes a `user_id` as input and returns the top-K most relevant items for that user.

---

## 📊 Available Datasets

### 1. Item Catalog
**Source:** `data/default_item_catalog.json` (5 items)

```json
[
  {
    "item_id": "item_001",
    "title": "Wireless Bluetooth Earbuds",
    "category": "electronics",
    "tags": ["wireless", "audio", "budget"],
    "price": 29.99,
    "description": "Compact earbuds with noise isolation and 24-hour battery case."
  },
  {
    "item_id": "item_002",
    "title": "The Martian",
    "category": "books",
    "tags": ["fiction", "sci-fi", "bestseller"],
    "price": 14.99,
    "description": "A stranded astronaut fights to survive on Mars."
  },
  {
    "item_id": "item_003",
    "title": "Python for Data Analysis",
    "category": "books",
    "tags": ["non-fiction", "programming", "data"],
    "price": 39.99,
    "description": "Practical guide to pandas, NumPy, and data workflows."
  },
  {
    "item_id": "item_004",
    "title": "Mechanical Keyboard",
    "category": "electronics",
    "tags": ["gaming", "peripherals", "mechanical"],
    "price": 89.99,
    "description": "Hot-swappable switches with RGB backlighting."
  },
  {
    "item_id": "item_005",
    "title": "Travel Guide: Amsterdam",
    "category": "travel",
    "tags": ["europe", "city-guide", "canals"],
    "price": 19.99,
    "description": "Neighborhoods, museums, and day trips around Amsterdam."
  }
]
```

**Item Schema:**
- `item_id` (string): Unique item identifier
- `title` (string): Display name
- `category` (string): Item category (electronics, books, travel, etc.)
- `tags` (array): Searchable metadata tags
- `price` (float): Item price
- `description` (string): Detailed description

---

### 2. User Profile Dataset
**Source:** `data/default_user_profile.json`

```json
{
  "user_id": "user_001",
  "segment": "returning_customer",
  "notes": "Interested in tech gadgets and sci-fi books."
}
```

**User Profile Schema:**
- `user_id` (string): Unique user identifier
- `segment` (string): User segment (returning_customer, new_customer, vip, etc.)
- `notes` (string): Behavioral or preference notes

---

### 3. User Interactions Dataset
**Source:** `data/default_llminput.json` → `target_user_interactions`

```json
[
  { "timestamp": "2026-06-01T10:15:00Z", "item_id": "item_001", "action": "view" },
  { "timestamp": "2026-06-01T10:16:30Z", "item_id": "item_001", "action": "click" },
  { "timestamp": "2026-06-03T09:05:00Z", "item_id": "item_001", "action": "purchase" },
  { "timestamp": "2026-06-10T18:40:00Z", "item_id": "item_002", "action": "view" },
  { "timestamp": "2026-06-10T18:41:00Z", "item_id": "item_002", "action": "like" }
]
```

**Interaction Schema:**
- `timestamp` (ISO8601): When the interaction occurred
- `item_id` (string): Which item
- `action` (string): Type of interaction (view, click, add_to_cart, purchase, like, dislike, rate, share, remove)
- `value` (optional): Rating or quantitative value
- `session_id` (optional): Session identifier
- `context` (optional): Additional context

---

### 4. Peer Users & Collaborative Filtering Data
**Source:** `data/default_llminput.json` → `other_users_interactions`

```json
[
  {
    "user_id": "user_002",
    "segment": "new_customer",
    "notes": "Browsing electronics and gaming gear.",
    "interactions": [
      { "timestamp": "2026-06-05T08:00:00Z", "item_id": "item_001", "action": "purchase" },
      { "timestamp": "2026-06-06T12:00:00Z", "item_id": "item_004", "action": "purchase" }
    ]
  },
  {
    "user_id": "user_003",
    "segment": "returning_customer",
    "notes": "Prefers books and programming content.",
    "interactions": [
      { "timestamp": "2026-06-08T10:30:00Z", "item_id": "item_002", "action": "purchase" },
      { "timestamp": "2026-06-11T09:20:00Z", "item_id": "item_003", "action": "purchase" }
    ]
  }
]
```

**Purpose:** Identify patterns from similar users (collaborative filtering).

---

## 🎯 Recommendation Engine Requirements

### Input
```python
{
  "user_id": "user_001",
  "top_k": 3,
  "exclude_categories": [],  # optional
  "min_price": 0,            # optional
  "max_price": 100,          # optional
  "diversify": True          # optional: return diverse item categories
}
```

### Output
```json
{
  "user_id": "user_001",
  "recommendations": [
    {
      "rank": 1,
      "item_id": "item_003",
      "title": "Python for Data Analysis",
      "category": "books",
      "score": 0.92,
      "reason": "User_003 (similar segment: returning_customer) purchased this after buying The Martian. Aligns with your interest in tech and data.",
      "signals": [
        "collaborative_filtering: user_003 purchased",
        "category_affinity: books (80% of likes)",
        "recency_bonus: recent peer purchase"
      ]
    },
    {
      "rank": 2,
      "item_id": "item_004",
      "title": "Mechanical Keyboard",
      "category": "electronics",
      "score": 0.78,
      "reason": "High-interest electronics for tech enthusiasts. Complements your wireless earbuds purchase.",
      "signals": [
        "category_preference: electronics",
        "complementary_purchase: pairs with item_001",
        "peer_adoption: user_002 purchased"
      ]
    },
    {
      "rank": 3,
      "item_id": "item_005",
      "title": "Travel Guide: Amsterdam",
      "category": "travel",
      "score": 0.65,
      "reason": "Exploratory recommendation: new category. Popular with returning customers; may appeal beyond tech.",
      "signals": [
        "exploration_bonus: new category",
        "segment_match: returning_customer segment"
      ]
    }
  ],
  "excluded_items": ["item_001", "item_002"],
  "excluded_reason": "Already purchased or liked",
  "generated_at": "2026-06-29T12:00:00Z"
}
```

---

## 8. llminput — Placeholder Reference

The `llminput` object contains all the input parameters for the recommendation engine. Here's a reference with **3 sample items** from the catalog:

```json
{
  "user_id": "user_001",
  "top_k": 3,
  "user_profile": {
    "user_id": "user_001",
    "segment": "returning_customer",
    "notes": "Interested in tech gadgets and sci-fi books."
  },
  "target_user_interactions": [
    { "timestamp": "2026-06-01T10:15:00Z", "item_id": "item_001", "action": "view" },
    { "timestamp": "2026-06-03T09:05:00Z", "item_id": "item_001", "action": "purchase" },
    { "timestamp": "2026-06-10T18:41:00Z", "item_id": "item_002", "action": "like" }
  ],
  "item_catalog": [
    {
      "item_id": "item_001",
      "title": "Wireless Bluetooth Earbuds",
      "category": "electronics",
      "tags": ["wireless", "audio", "budget"],
      "price": 29.99,
      "description": "Compact earbuds with noise isolation and 24-hour battery case."
    },
    {
      "item_id": "item_002",
      "title": "The Martian",
      "category": "books",
      "tags": ["fiction", "sci-fi", "bestseller"],
      "price": 14.99,
      "description": "A stranded astronaut fights to survive on Mars."
    },
    {
      "item_id": "item_003",
      "title": "Python for Data Analysis",
      "category": "books",
      "tags": ["non-fiction", "programming", "data"],
      "price": 39.99,
      "description": "Practical guide to pandas, NumPy, and data workflows."
    }
  ],
  "other_users_interactions": [
    {
      "user_id": "user_002",
      "segment": "new_customer",
      "notes": "Browsing electronics and gaming gear.",
      "interactions": [
        { "timestamp": "2026-06-05T08:00:00Z", "item_id": "item_001", "action": "purchase" }
      ]
    },
    {
      "user_id": "user_003",
      "segment": "returning_customer",
      "notes": "Prefers books and programming content.",
      "interactions": [
        { "timestamp": "2026-06-08T10:30:00Z", "item_id": "item_002", "action": "purchase" },
        { "timestamp": "2026-06-11T09:20:00Z", "item_id": "item_003", "action": "purchase" }
      ]
    }
  ],
  "action_weights": "| Action | Weight | Meaning |\n|--------|--------|----------|\n| purchase | 1.0 | Strong positive signal |\n| like | 0.75 | Explicit preference |\n| click | 0.50 | Interest signal |\n| view | 0.30 | Weak signal |\n| dislike | -1.0 | Hard negative |",
  "constraints": "- Exclude items already purchased or disliked by the Target User.\n- Prefer items in categories the user engaged with recently.\n- Balance exploitation with one exploratory item.",
  "allowed_actions": "view, click, add_to_cart, purchase, like, dislike, rate, share, remove",
  "llm_chat": "",
  "generated_at": "2026-06-29T12:00:00Z"
}
```

**Key Fields:**
- `user_profile`: Target user's demographic and preference data
- `target_user_interactions`: Historical actions by the target user (chronological)
- `item_catalog`: List of available items (shown: 3 sample items; full catalog may contain more)
- `other_users_interactions`: Peer user data for collaborative filtering
- `action_weights`: Scoring weights for each interaction type
- `constraints`: Business rules and exclusion criteria
- `top_k`: Number of recommendations to generate

---

## ⚖️ Action Weights (Scoring Signals)

Use these weights to score user-item interactions:

| Action | Weight | Meaning |
|--------|--------|---------|
| **purchase** | 1.0 | Strong positive signal — highest intent |
| **add_to_cart** | 0.85 | High intent to buy |
| **like** | 0.75 | Explicit positive preference |
| **rate** | 0.70 | Use rating value (1-5 stars) |
| **share** | 0.65 | Social endorsement |
| **click** | 0.50 | Interest signal |
| **view** | 0.30 | Weak signal — just browsing |
| **remove** | -0.5 | Negative signal — removed from cart |
| **dislike** | -1.0 | Hard negative — exclude from recommendations |

---

## 📈 Scoring Algorithm (Content + Collaborative Filtering)

### Step 1: Content-Based Scoring
For each candidate item, calculate relevance based on user's history:

```
content_score(item, user) = 
    category_affinity(item.category, user) * 0.4 +
    tag_similarity(item.tags, user.preferences) * 0.3 +
    price_fit(item.price, user.budget) * 0.2 +
    recency_bonus(user.interactions) * 0.1
```

**Category Affinity:** What % of user's actions are in this category?
**Tag Similarity:** How much do item tags overlap with user's preferred tags?
**Price Fit:** How does price compare to user's historical purchases?
**Recency Bonus:** Recent interactions weighted higher than old ones.

### Step 2: Collaborative Filtering Scoring
Find similar users and boost score if they purchased/liked this item:

```
collab_score(item, user) =
    sum(
        similarity(user, peer_user) * 
        peer_user.action_weight(item) 
        for peer_user in similar_users
    ) / num_similar_users
```

**Similarity Metric:** Users are similar if they:
- Share the same segment
- Have purchased from the same categories
- Have rated items in common

### Step 3: Final Score
```
final_score(item, user) = 
    (content_score * 0.6) + (collab_score * 0.4)
```

---

## 🚫 Exclusion Rules

**Never recommend items where:**
1. User has already **purchased** the item
2. User has **disliked** or **removed** the item
3. Item is out of stock (if available data)
4. Item price exceeds user's max_price threshold
5. Item category is in exclude_categories list

**For user_001 example:**
- ❌ Exclude `item_001` (purchased)
- ❌ Exclude `item_002` (liked, but check if already purchased)
- ✅ Recommend from: item_003, item_004, item_005

---

## 🎨 Diversity & Exploration Strategy

**Exploitation vs. Exploration (80/20 rule):**
- **80%** of recommendations: High-confidence matches (score > 0.7)
- **20%** of recommendations: Exploration — new categories or lower-confidence items

**For top_k=3:**
- Rank 1: Highest score (exploitation)
- Rank 2: High score, different category if possible (exploitation)
- Rank 3: Exploratory item, new category (exploration)

---

## 🔍 Case Study: Generating Recommendations for `user_001`

### Given Data:
- **User:** user_001 (returning_customer, tech gadgets + sci-fi)
- **History:** Purchased earbuds (electronics), viewed/liked Martian (sci-fi book)
- **Peers:** user_002 (electronics), user_003 (books + programming)
- **Catalog:** 3 sample items (shown below)

**Sample Item Catalog:**
```json
[
  {
    "item_id": "item_001",
    "title": "Wireless Bluetooth Earbuds",
    "category": "electronics",
    "tags": ["wireless", "audio", "budget"],
    "price": 29.99,
    "description": "Compact earbuds with noise isolation and 24-hour battery case."
  },
  {
    "item_id": "item_002",
    "title": "The Martian",
    "category": "books",
    "tags": ["fiction", "sci-fi", "bestseller"],
    "price": 14.99,
    "description": "A stranded astronaut fights to survive on Mars."
  },
  {
    "item_id": "item_003",
    "title": "Python for Data Analysis",
    "category": "books",
    "tags": ["non-fiction", "programming", "data"],
    "price": 39.99,
    "description": "Practical guide to pandas, NumPy, and data workflows."
  }
]
```

### Algorithm Execution:

| Item | Category | Content Score | Collab Score | Final Score | Rank | Include? |
|------|----------|---|---|---|---|---|
| item_001 | electronics | 0.85 | 0.80 | 0.825 | - | ❌ Purchased |
| item_002 | books | 0.88 | 0.75 | 0.825 | - | ⚠️ Already liked |
| item_003 | books | 0.89 | 0.92 | 0.905 | **1** | ✅ **Recommend** |

**Final Recommendations:** item_003 (top recommendation)

---

## 💻 Implementation Checklist

When implementing the engine, ensure:

- [ ] **Load & parse** item catalog from JSON
- [ ] **Retrieve** user profile by `user_id`
- [ ] **Fetch** user interactions (timestamp, action, item_id)
- [ ] **Identify** peer users with similar segments/preferences
- [ ] **Calculate** content-based scores using category/tag affinity
- [ ] **Calculate** collaborative filtering scores from peer behavior
- [ ] **Combine** scores (60% content + 40% collab)
- [ ] **Apply** exclusion rules (purchased, disliked, etc.)
- [ ] **Rank** by final score
- [ ] **Select** top_k items, ensuring diversity
- [ ] **Generate** human-readable reason for each recommendation
- [ ] **Return** structured JSON with scores, signals, and reasoning

---

## 🎓 Key Metrics to Track

For each recommendation, calculate & return:
1. **Score:** 0.0 to 1.0 (confidence)
2. **Rank:** Position in top-k list
3. **Signals:** Which factors contributed (content, collab, diversity, etc.)
4. **Reason:** Human-readable explanation for user
5. **Confidence:** High (>0.8), Medium (0.5-0.8), Low (<0.5)

---

## 📝 Notes

- **Timestamps matter:** Weight recent interactions more heavily (e.g., exponential decay over time)
- **Sparse data:** If user has few interactions, rely more on item popularity and category priors
- **Cold start:** New users with no history → use segment-based recommendations
- **Edge cases:** Handle users with no interaction data, items with no peer interest, etc.
- **Explainability:** Always provide a reason so users understand why items were recommended

---

**Your recommendation engine should accept `user_id` and return personalized, explainable recommendations using the datasets above.**
