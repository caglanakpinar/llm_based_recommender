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

{action_weights}

### Rules
1. Weight **recent** interactions more than older ones.
2. Treat repeated actions on the same item as stronger preference.
3. Use **Other Users** to find collaborative patterns (users who liked similar items).
4. If data is sparse, rely more on item metadata and popular items in the same category.
5. Return exactly **{top_k}** recommendations unless stated otherwise.
6. For each recommendation, give a **short, human-readable reason** tied to specific interactions or similar users.

---

## 2. Item Catalog

Built from parquet using column mapping:

```json
{item_catalog}
```

**Item schema — parquet column mapping (each element):**

```json
{
  "item_id": "{item_id column in parquet}",
  "title": "{title column in parquet}",
  "category": "{category column in parquet}",
  "tags": "{tags column in parquet}",
  "price": "{price column in parquet}",
  "description": "{description column in parquet}"
}
```

---

## 3. User Profile

Built from parquet using column mapping for the **Target User**:

```json
{user_profile}
```

**Profile schema — parquet column mapping:**

```json
{
  "user_id": "{user_id column in parquet}",
  "segment": "{segment column in parquet, optional}",
  "notes": "{notes column in parquet, optional}"
}
```

---

## 4. Target User — Item Interactions

Chronological list of what **this user** did, **when**, and on **which item**.

```json
{target_user_interactions}
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

**Allowed `action` values:** `{allowed_actions}`

**Optional fields:** `value`, `session_id`, `context` — omit when not applicable.

---

## 5. Other Users — Item Interactions

Interactions from **other users**. Used for collaborative filtering and “users like you also liked …” reasoning.

```json
{other_users_interactions}
```

**Other-user schema (each element) — profile fields from the same parquet column mapping as Section 3:**

```json
{
  "user_id": "{user_id column in parquet}",
  "segment": "{segment column in parquet, optional}",
  "notes": "{notes column in parquet, optional}",
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
- the **User Profile** (Section 3),
- the **Target User interactions** (Section 4),
- and **Other Users' interactions** (Section 5),

generate **{top_k}** personalized recommendations for the user described in **Section 3** (`user_profile.user_id`).

**Constraints:**

{constraints}

---

## 7. Expected LLM Response Format

The model **must** respond with valid JSON only (no markdown wrapper):

```json
{
  "user_id": "{user_id}",
  "generated_at": "{generated_at}",
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
| `user_id` | string | Must match `user_profile.user_id` from Section 3 |
| `generated_at` | ISO 8601 | `{generated_at}` — when recommendations were produced |
| `recommendations` | array | Ordered list, best first; length = `{top_k}` |
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
| `{top_k}` | integer | Number of recommendations to return |
| `{generated_at}` | ISO 8601 | Timestamp for the response (optional pre-fill) |
| `{action_weights}` | markdown table or JSON | Action → weight mapping |
| `{item_catalog}` | JSON array | Full item catalog (Section 2) |
| `{user_profile}` | JSON object | Target user profile (Section 3) |
| `{target_user_interactions}` | JSON array | Target user event history (Section 4) |
| `{other_users_interactions}` | JSON array | Peer user event histories (Section 5) |
| `{constraints}` | markdown list | Recommendation constraints (Section 6) |
| `{allowed_actions}` | comma-separated string | Valid action enum values |
| `{llm_chat}` | markdown | Additional instructions appended at the end |

**Parquet column mapping fields (used when building Sections 2–3):**

| Field | Parquet column | Required |
|-------|----------------|----------|
| `user_id` | user identifier | yes |
| `segment` | user segment / cohort | no |
| `notes` | free-text user context | no |
| `item_id` | item identifier | yes |
| `title` | item title | yes |
| `category` | item category | yes |
| `tags` | item tags (comma-separated or JSON array) | no |
| `price` | item price | no |
| `description` | item description | no |

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
| `{segment}` | string | User segment / cohort |
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

**Your answer:** Recommendations for the user in **Section 3** as defined in Section 7.

---

## 9. Additional Instructions

{llm_chat}
