# LLM Recommender System — Prompt Template

> Send this document to the LLM after filling all `{placeholder}` values via llminput. The model returns personalized recommendations for the **Target User**.

---

## 1. Engine Description

You are a **recommendation engine**. Your job is to suggest items the **Target User** is most likely to engage with next.

### Goals
- Maximize relevance to the Target User's tastes and intent.
- Use recency, frequency, and action strength (e.g. purchase > view).
- Learn from **similar users** without overfitting to popularity alone.
- Prefer diverse, non-redundant recommendations when scores are close.
- Never recommend items the Target User has already **purchased** or **explicitly disliked**, unless asked.

### Action weights (stronger = higher intent)

| Action | Weight | Meaning |
|--------|--------|----------|
| purchase | 1.0 | Strong positive signal |
| add_to_cart | 0.85 | High intent |
| like | 0.75 | Explicit preference |
| rate | 0.70 | Use rating value |
| click | 0.50 | Interest signal |
| view | 0.30 | Weak signal |
| share | 0.65 | Social endorsement |
| dislike | -1.0 | Hard negative |
| remove | -0.5 | Negative signal |

### Rules
1. Weight **recent** interactions more than older ones.
2. Treat repeated actions on the same item as stronger preference.
3. Use **Other Users** to find collaborative patterns (users who liked similar items).
4. If data is sparse, rely more on item metadata and popular items in the same category.
5. Return exactly **5** recommendations unless stated otherwise.
6. For each recommendation, give a **short, human-readable reason** tied to specific interactions or similar users.

---

## 2. Item Catalog

```json
[
  {
    "item_id": "item_001",
    "title": "Wireless Bluetooth Earbuds",
    "category": "electronics",
    "tags": [
      "wireless",
      "audio",
      "budget"
    ],
    "price": 29.99,
    "description": "Compact earbuds with noise isolation and 24-hour battery case."
  },
  {
    "item_id": "item_002",
    "title": "The Martian",
    "category": "books",
    "tags": [
      "fiction",
      "sci-fi",
      "bestseller"
    ],
    "price": 14.99,
    "description": "A stranded astronaut fights to survive on Mars."
  },
  {
    "item_id": "item_003",
    "title": "Python for Data Analysis",
    "category": "books",
    "tags": [
      "non-fiction",
      "programming",
      "data"
    ],
    "price": 39.99,
    "description": "Practical guide to pandas, NumPy, and data workflows."
  },
  {
    "item_id": "item_004",
    "title": "Mechanical Keyboard",
    "category": "electronics",
    "tags": [
      "gaming",
      "peripherals",
      "mechanical"
    ],
    "price": 89.99,
    "description": "Hot-swappable switches with RGB backlighting."
  },
  {
    "item_id": "item_005",
    "title": "Travel Guide: Amsterdam",
    "category": "travel",
    "tags": [
      "europe",
      "city-guide",
      "canals"
    ],
    "price": 19.99,
    "description": "Neighborhoods, museums, and day trips around Amsterdam."
  }
]
```

**Item schema (each element):**

```json
{
  "item_id": "{item_id}",
  "title": "{title}",
  "category": "{category}",
  "tags": ["{tag}"],
  "price": {price},
  "description": "{description}"
}
```

---

## 3. Target User Profile

```json
{
  "user_id": "user_001",
  "display_name": "Alex",
  "segment": "returning_customer",
  "preferences": {
    "categories": [
      "electronics",
      "books"
    ],
    "price_range": {
      "min": 10,
      "max": 100,
      "currency": "USD"
    }
  },
  "notes": "Interested in tech gadgets and sci-fi books."
}
```

**Profile schema:**

```json
{
  "user_id": "{user_id}",
  "display_name": "{display_name}",
  "segment": "{segment}",
  "preferences": {
    "categories": ["{category}"],
    "price_range": {
      "min": {price_min},
      "max": {price_max},
      "currency": "{currency}"
    }
  },
  "notes": "{notes}"
}
```

---

## 4. Target User — Item Interactions

Chronological list of what **this user** did, **when**, and on **which item**.

```json
[
  {
    "timestamp": "2026-06-01T10:15:00Z",
    "item_id": "item_001",
    "action": "view"
  },
  {
    "timestamp": "2026-06-01T10:16:30Z",
    "item_id": "item_001",
    "action": "click"
  },
  {
    "timestamp": "2026-06-03T09:05:00Z",
    "item_id": "item_001",
    "action": "purchase"
  },
  {
    "timestamp": "2026-06-10T18:40:00Z",
    "item_id": "item_002",
    "action": "view"
  },
  {
    "timestamp": "2026-06-10T18:41:00Z",
    "item_id": "item_002",
    "action": "like"
  }
]
```

**Interaction schema (each element):**

```json
{
  "timestamp": "{timestamp}",
  "item_id": "{item_id}",
  "action": "{action}",
  "value": {value},
  "session_id": "{session_id}",
  "context": "{context}"
}
```

**Allowed `action` values:** `view, click, add_to_cart, purchase, like, dislike, rate, share, remove`

**Optional fields:** `value`, `session_id`, `context` — omit when not applicable.

---

## 5. Other Users — Item Interactions

Interactions from **other users**. Used for collaborative filtering and “users like you also liked …” reasoning.

```json
[
  {
    "user_id": "user_002",
    "interactions": [
      {
        "timestamp": "2026-06-05T08:00:00Z",
        "item_id": "item_001",
        "action": "purchase"
      },
      {
        "timestamp": "2026-06-06T12:00:00Z",
        "item_id": "item_004",
        "action": "purchase"
      }
    ]
  },
  {
    "user_id": "user_003",
    "interactions": [
      {
        "timestamp": "2026-06-08T10:30:00Z",
        "item_id": "item_002",
        "action": "purchase"
      },
      {
        "timestamp": "2026-06-11T09:20:00Z",
        "item_id": "item_003",
        "action": "purchase"
      }
    ]
  }
]
```

**Other-user schema (each element):**

```json
{
  "user_id": "{user_id}",
  "interactions": [
    {
      "timestamp": "{timestamp}",
      "item_id": "{item_id}",
      "action": "{action}",
      "value": {value},
      "session_id": "{session_id}",
      "context": "{context}"
    }
  ]
}
```

---

## 6. Request

Based on:
- the **Engine Description** (Section 1),
- the **Item Catalog** (Section 2),
- the **Target User Profile** (Section 3),
- the **Target User interactions** (Section 4),
- and **Other Users' interactions** (Section 5),

generate **5** personalized recommendations for user **`user_999`**.

**Constraints:**

- Exclude items already purchased or disliked by the Target User.
- Prefer items in categories the user engaged with recently.
- Balance exploitation with one exploratory item.

---

## 7. Expected LLM Response Format

The model **must** respond with valid JSON only (no markdown wrapper):

```json
{
  "target_user_id": "user_999",
  "generated_at": "2026-06-27T12:00:00Z",
  "recommendations": [
    {
      "rank": {rank},
      "item_id": "{item_id}",
      "score": {score},
      "reason": "{reason}"
    }
  ],
  "summary": "{summary}"
}
```

### Response field definitions

| Field | Type | Description |
|-------|------|-------------|
| `target_user_id` | string | Must match `user_999` from Section 3 |
| `generated_at` | ISO 8601 | `2026-06-27T12:00:00Z` — when recommendations were produced |
| `recommendations` | array | Ordered list, best first; length = `5` |
| `recommendations[].rank` | integer | `{rank}` — 1 = top recommendation |
| `recommendations[].item_id` | string | `{item_id}` from catalog or interaction data |
| `recommendations[].score` | float | `{score}` — 0.0–1.0 confidence / relevance |
| `recommendations[].reason` | string | `{reason}` — brief justification citing user or peer behavior |
| `summary` | string | `{summary}` — high-level explanation for the Target User or downstream UI |

---

## 8. llminput — Placeholder Reference

All values below are injected at runtime. Replace every `{key}` in this document.

| Placeholder | Type | Description |
|-------------|------|-------------|
| `5` | integer | Number of recommendations to return |
| `user_999` | string | User ID to recommend for |
| `2026-06-27T12:00:00Z` | ISO 8601 | Timestamp for the response (optional pre-fill) |
| `| Action | Weight | Meaning |
|--------|--------|----------|
| purchase | 1.0 | Strong positive signal |
| add_to_cart | 0.85 | High intent |
| like | 0.75 | Explicit preference |
| rate | 0.70 | Use rating value |
| click | 0.50 | Interest signal |
| view | 0.30 | Weak signal |
| share | 0.65 | Social endorsement |
| dislike | -1.0 | Hard negative |
| remove | -0.5 | Negative signal |` | markdown table or JSON | Action → weight mapping |
| `[
  {
    "item_id": "item_001",
    "title": "Wireless Bluetooth Earbuds",
    "category": "electronics",
    "tags": [
      "wireless",
      "audio",
      "budget"
    ],
    "price": 29.99,
    "description": "Compact earbuds with noise isolation and 24-hour battery case."
  },
  {
    "item_id": "item_002",
    "title": "The Martian",
    "category": "books",
    "tags": [
      "fiction",
      "sci-fi",
      "bestseller"
    ],
    "price": 14.99,
    "description": "A stranded astronaut fights to survive on Mars."
  },
  {
    "item_id": "item_003",
    "title": "Python for Data Analysis",
    "category": "books",
    "tags": [
      "non-fiction",
      "programming",
      "data"
    ],
    "price": 39.99,
    "description": "Practical guide to pandas, NumPy, and data workflows."
  },
  {
    "item_id": "item_004",
    "title": "Mechanical Keyboard",
    "category": "electronics",
    "tags": [
      "gaming",
      "peripherals",
      "mechanical"
    ],
    "price": 89.99,
    "description": "Hot-swappable switches with RGB backlighting."
  },
  {
    "item_id": "item_005",
    "title": "Travel Guide: Amsterdam",
    "category": "travel",
    "tags": [
      "europe",
      "city-guide",
      "canals"
    ],
    "price": 19.99,
    "description": "Neighborhoods, museums, and day trips around Amsterdam."
  }
]` | JSON array | Full item catalog (Section 2) |
| `{
  "user_id": "user_001",
  "display_name": "Alex",
  "segment": "returning_customer",
  "preferences": {
    "categories": [
      "electronics",
      "books"
    ],
    "price_range": {
      "min": 10,
      "max": 100,
      "currency": "USD"
    }
  },
  "notes": "Interested in tech gadgets and sci-fi books."
}` | JSON object | Target user profile (Section 3) |
| `[
  {
    "timestamp": "2026-06-01T10:15:00Z",
    "item_id": "item_001",
    "action": "view"
  },
  {
    "timestamp": "2026-06-01T10:16:30Z",
    "item_id": "item_001",
    "action": "click"
  },
  {
    "timestamp": "2026-06-03T09:05:00Z",
    "item_id": "item_001",
    "action": "purchase"
  },
  {
    "timestamp": "2026-06-10T18:40:00Z",
    "item_id": "item_002",
    "action": "view"
  },
  {
    "timestamp": "2026-06-10T18:41:00Z",
    "item_id": "item_002",
    "action": "like"
  }
]` | JSON array | Target user event history (Section 4) |
| `[
  {
    "user_id": "user_002",
    "interactions": [
      {
        "timestamp": "2026-06-05T08:00:00Z",
        "item_id": "item_001",
        "action": "purchase"
      },
      {
        "timestamp": "2026-06-06T12:00:00Z",
        "item_id": "item_004",
        "action": "purchase"
      }
    ]
  },
  {
    "user_id": "user_003",
    "interactions": [
      {
        "timestamp": "2026-06-08T10:30:00Z",
        "item_id": "item_002",
        "action": "purchase"
      },
      {
        "timestamp": "2026-06-11T09:20:00Z",
        "item_id": "item_003",
        "action": "purchase"
      }
    ]
  }
]` | JSON array | Peer user event histories (Section 5) |
| `- Exclude items already purchased or disliked by the Target User.
- Prefer items in categories the user engaged with recently.
- Balance exploitation with one exploratory item.` | markdown list | Recommendation constraints (Section 6) |
| `view, click, add_to_cart, purchase, like, dislike, rate, share, remove` | comma-separated string | Valid action enum values |

**Per-record fields (used when building JSON arrays):**

| Placeholder | Type | Description |
|-------------|------|-------------|
| `{item_id}` | string | Item identifier |
| `{title}` | string | Item title |
| `{category}` | string | Item or preference category |
| `{tag}` | string | Item tag (repeat in array as needed) |
| `{price}` | number | Item price |
| `{description}` | string | Item description |
| `{user_id}` | string | User identifier |
| `{display_name}` | string | User display name |
| `{segment}` | string | User segment / cohort |
| `{price_min}` | number | Min preferred price |
| `{price_max}` | number | Max preferred price |
| `{currency}` | string | Currency code (e.g. USD) |
| `{notes}` | string | Free-text user context |
| `{timestamp}` | ISO 8601 | Event time |
| `{action}` | string | Interaction type |
| `{value}` | number | Rating or custom score (omit if N/A) |
| `{session_id}` | string | Session grouping ID |
| `{context}` | string | Event source (search, homepage, etc.) |
| `{rank}` | integer | Recommendation rank |
| `{score}` | float | Recommendation score 0.0–1.0 |
| `{reason}` | string | Why this item was recommended |
| `{summary}` | string | Overall recommendation strategy sentence |

**Your answer:** Recommendations for **`user_999`** as defined in Section 7.
