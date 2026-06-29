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
5. Return exactly **3** recommendations unless stated otherwise.
6. For each recommendation, give a **short, human-readable reason** tied to specific interactions or similar users.

---

## 2. Item Catalog

Built from parquet using column mapping:

```json
[
  {
    "item_id": "item_0001",
    "title": "Product item_0001",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 60.2,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0002",
    "title": "Product item_0002",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 182.38,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0003",
    "title": "Product item_0003",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 210.16,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0004",
    "title": "Product item_0004",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 24.79,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0005",
    "title": "Product item_0005",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 221.11,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0006",
    "title": "Product item_0006",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 200.39,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0007",
    "title": "Product item_0007",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 218.07,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0008",
    "title": "Product item_0008",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 176.11,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0009",
    "title": "Product item_0009",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 170.17,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0010",
    "title": "Product item_0010",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 133.65,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0011",
    "title": "Product item_0011",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 164.93,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0012",
    "title": "Product item_0012",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 230.18,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0013",
    "title": "Product item_0013",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 29.63,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0014",
    "title": "Product item_0014",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 82.36,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0015",
    "title": "Product item_0015",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 22.75,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0016",
    "title": "Product item_0016",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 236.48,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0017",
    "title": "Product item_0017",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 114.09,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0018",
    "title": "Product item_0018",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 170.46,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0019",
    "title": "Product item_0019",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 211.07,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0020",
    "title": "Product item_0020",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 247.8,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0021",
    "title": "Product item_0021",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 44.64,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0022",
    "title": "Product item_0022",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 200.88,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0023",
    "title": "Product item_0023",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 222.42,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0024",
    "title": "Product item_0024",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 16.97,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0025",
    "title": "Product item_0025",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 189.75,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0026",
    "title": "Product item_0026",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 91.91,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0027",
    "title": "Product item_0027",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 163.91,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0028",
    "title": "Product item_0028",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 130.94,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0029",
    "title": "Product item_0029",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 68.64,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0030",
    "title": "Product item_0030",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 38.13,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0031",
    "title": "Product item_0031",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 33.65,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0032",
    "title": "Product item_0032",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 30.31,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0033",
    "title": "Product item_0033",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 161.26,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0034",
    "title": "Product item_0034",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 91.1,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0035",
    "title": "Product item_0035",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 58.08,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0036",
    "title": "Product item_0036",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 174.51,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0037",
    "title": "Product item_0037",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 173.28,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0038",
    "title": "Product item_0038",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 137.1,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0039",
    "title": "Product item_0039",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 234.06,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0040",
    "title": "Product item_0040",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 99.43,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0041",
    "title": "Product item_0041",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 209.79,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0042",
    "title": "Product item_0042",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 23.41,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0043",
    "title": "Product item_0043",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 186.26,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0044",
    "title": "Product item_0044",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 247.17,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0045",
    "title": "Product item_0045",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 212.42,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0046",
    "title": "Product item_0046",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 147.02,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0047",
    "title": "Product item_0047",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 158.56,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0048",
    "title": "Product item_0048",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 91.56,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0049",
    "title": "Product item_0049",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 171.47,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0050",
    "title": "Product item_0050",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 193.75,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0051",
    "title": "Product item_0051",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 249.57,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0052",
    "title": "Product item_0052",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 125.65,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0053",
    "title": "Product item_0053",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 20.02,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0054",
    "title": "Product item_0054",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 150.99,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0055",
    "title": "Product item_0055",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 34.16,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0056",
    "title": "Product item_0056",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 64.97,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0057",
    "title": "Product item_0057",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 85.55,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0058",
    "title": "Product item_0058",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 184.1,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0059",
    "title": "Product item_0059",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 50.03,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0060",
    "title": "Product item_0060",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 45.61,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0061",
    "title": "Product item_0061",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 16.02,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0062",
    "title": "Product item_0062",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 218.94,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0063",
    "title": "Product item_0063",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 174.91,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0064",
    "title": "Product item_0064",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 14.79,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0065",
    "title": "Product item_0065",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 11.93,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0066",
    "title": "Product item_0066",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 141.88,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0067",
    "title": "Product item_0067",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 74.22,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0068",
    "title": "Product item_0068",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 151.2,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0069",
    "title": "Product item_0069",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 111.48,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0070",
    "title": "Product item_0070",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 231.21,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0071",
    "title": "Product item_0071",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 75.51,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0072",
    "title": "Product item_0072",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 171.39,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0073",
    "title": "Product item_0073",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 18.24,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0074",
    "title": "Product item_0074",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 239.19,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0075",
    "title": "Product item_0075",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 245.29,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0076",
    "title": "Product item_0076",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 186.07,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0077",
    "title": "Product item_0077",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 118.98,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0078",
    "title": "Product item_0078",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 58.9,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0079",
    "title": "Product item_0079",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 204.07,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0080",
    "title": "Product item_0080",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 185.41,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0081",
    "title": "Product item_0081",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 59.38,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0082",
    "title": "Product item_0082",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 115.55,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0083",
    "title": "Product item_0083",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 118.39,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0084",
    "title": "Product item_0084",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 207.47,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0085",
    "title": "Product item_0085",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 172.54,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0086",
    "title": "Product item_0086",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 162.58,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0087",
    "title": "Product item_0087",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 164.21,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0088",
    "title": "Product item_0088",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 71.27,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0089",
    "title": "Product item_0089",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 96.07,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0090",
    "title": "Product item_0090",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 16.19,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0091",
    "title": "Product item_0091",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 197.8,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0092",
    "title": "Product item_0092",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 111.08,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0093",
    "title": "Product item_0093",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 112.37,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0094",
    "title": "Product item_0094",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 65.11,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0095",
    "title": "Product item_0095",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 56.31,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0096",
    "title": "Product item_0096",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 149.02,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0097",
    "title": "Product item_0097",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 61.98,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0098",
    "title": "Product item_0098",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 20.73,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0099",
    "title": "Product item_0099",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 193.04,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0100",
    "title": "Product item_0100",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 97.24,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0101",
    "title": "Product item_0101",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 98.16,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0102",
    "title": "Product item_0102",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 110.67,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0103",
    "title": "Product item_0103",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 204.69,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0104",
    "title": "Product item_0104",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 83.61,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0105",
    "title": "Product item_0105",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 190.72,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0106",
    "title": "Product item_0106",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 235.19,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0107",
    "title": "Product item_0107",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 166.91,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0108",
    "title": "Product item_0108",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 110.76,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0109",
    "title": "Product item_0109",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 188.21,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0110",
    "title": "Product item_0110",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 117.18,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0111",
    "title": "Product item_0111",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 106.53,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0112",
    "title": "Product item_0112",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 133.69,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0113",
    "title": "Product item_0113",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 82.84,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0114",
    "title": "Product item_0114",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 30.54,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0115",
    "title": "Product item_0115",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 106.64,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0116",
    "title": "Product item_0116",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 35.34,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0117",
    "title": "Product item_0117",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 105.56,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0118",
    "title": "Product item_0118",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 141.45,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0119",
    "title": "Product item_0119",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 233.02,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0120",
    "title": "Product item_0120",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 13.87,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0121",
    "title": "Product item_0121",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 157.26,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0122",
    "title": "Product item_0122",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 117.15,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0123",
    "title": "Product item_0123",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 9.96,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0124",
    "title": "Product item_0124",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 66.14,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0125",
    "title": "Product item_0125",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 34.03,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0126",
    "title": "Product item_0126",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 58.11,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0127",
    "title": "Product item_0127",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 72.27,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0128",
    "title": "Product item_0128",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 103.32,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0129",
    "title": "Product item_0129",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 134.61,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0130",
    "title": "Product item_0130",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 5.24,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0131",
    "title": "Product item_0131",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 44.66,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0132",
    "title": "Product item_0132",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 27.92,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0133",
    "title": "Product item_0133",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 139.67,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0134",
    "title": "Product item_0134",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 85.62,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0135",
    "title": "Product item_0135",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 139.36,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0136",
    "title": "Product item_0136",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 59.35,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0137",
    "title": "Product item_0137",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 10.94,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0138",
    "title": "Product item_0138",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 235.63,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0139",
    "title": "Product item_0139",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 31.54,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0140",
    "title": "Product item_0140",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 36.08,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0141",
    "title": "Product item_0141",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 40.9,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0142",
    "title": "Product item_0142",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 245.41,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0143",
    "title": "Product item_0143",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 156.33,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0144",
    "title": "Product item_0144",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 154.52,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0145",
    "title": "Product item_0145",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 246.74,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0146",
    "title": "Product item_0146",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 26.67,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0147",
    "title": "Product item_0147",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 243.75,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0148",
    "title": "Product item_0148",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 54.98,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0149",
    "title": "Product item_0149",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 157.65,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0150",
    "title": "Product item_0150",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 107.92,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0151",
    "title": "Product item_0151",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag3"
    ],
    "price": 99.07,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0152",
    "title": "Product item_0152",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 233.15,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0153",
    "title": "Product item_0153",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 41.56,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0154",
    "title": "Product item_0154",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 249.28,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0155",
    "title": "Product item_0155",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 97.89,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0156",
    "title": "Product item_0156",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 163.95,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0157",
    "title": "Product item_0157",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 242.14,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0158",
    "title": "Product item_0158",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag3"
    ],
    "price": 19.11,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0159",
    "title": "Product item_0159",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 231.1,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0160",
    "title": "Product item_0160",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 144.11,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0161",
    "title": "Product item_0161",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 177.58,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0162",
    "title": "Product item_0162",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 41.55,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0163",
    "title": "Product item_0163",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 135.68,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0164",
    "title": "Product item_0164",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 170.01,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0165",
    "title": "Product item_0165",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 116.78,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0166",
    "title": "Product item_0166",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 42.97,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0167",
    "title": "Product item_0167",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 64.81,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0168",
    "title": "Product item_0168",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 214.33,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0169",
    "title": "Product item_0169",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 13.45,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0170",
    "title": "Product item_0170",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 178.67,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0171",
    "title": "Product item_0171",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 249.85,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0172",
    "title": "Product item_0172",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 6.83,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0173",
    "title": "Product item_0173",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 44.46,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0174",
    "title": "Product item_0174",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 170.17,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0175",
    "title": "Product item_0175",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 19.7,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0176",
    "title": "Product item_0176",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 149.5,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0177",
    "title": "Product item_0177",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 172.58,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0178",
    "title": "Product item_0178",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 149.1,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0179",
    "title": "Product item_0179",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag3"
    ],
    "price": 94.47,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0180",
    "title": "Product item_0180",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 175.83,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0181",
    "title": "Product item_0181",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 150.6,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0182",
    "title": "Product item_0182",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 55.87,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0183",
    "title": "Product item_0183",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 142.97,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0184",
    "title": "Product item_0184",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 221.07,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0185",
    "title": "Product item_0185",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 168.37,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0186",
    "title": "Product item_0186",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 120.59,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0187",
    "title": "Product item_0187",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 83.79,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0188",
    "title": "Product item_0188",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 26.04,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0189",
    "title": "Product item_0189",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 64.93,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0190",
    "title": "Product item_0190",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 7.74,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0191",
    "title": "Product item_0191",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 246.72,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0192",
    "title": "Product item_0192",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 59.51,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0193",
    "title": "Product item_0193",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 204.64,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0194",
    "title": "Product item_0194",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 46.32,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0195",
    "title": "Product item_0195",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 74.48,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0196",
    "title": "Product item_0196",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 107.48,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0197",
    "title": "Product item_0197",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 84.48,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0198",
    "title": "Product item_0198",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 200.6,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0199",
    "title": "Product item_0199",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 239.63,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0200",
    "title": "Product item_0200",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 219.45,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0201",
    "title": "Product item_0201",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 42.63,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0202",
    "title": "Product item_0202",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 195.29,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0203",
    "title": "Product item_0203",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 41.25,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0204",
    "title": "Product item_0204",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 228.78,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0205",
    "title": "Product item_0205",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 131.88,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0206",
    "title": "Product item_0206",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 87.41,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0207",
    "title": "Product item_0207",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 43.47,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0208",
    "title": "Product item_0208",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 7.89,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0209",
    "title": "Product item_0209",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 65.75,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0210",
    "title": "Product item_0210",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 228.93,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0211",
    "title": "Product item_0211",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 200.33,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0212",
    "title": "Product item_0212",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 242.6,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0213",
    "title": "Product item_0213",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 29.69,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0214",
    "title": "Product item_0214",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 150.08,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0215",
    "title": "Product item_0215",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 68,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0216",
    "title": "Product item_0216",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 138.02,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0217",
    "title": "Product item_0217",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 83.51,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0218",
    "title": "Product item_0218",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 74.45,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0219",
    "title": "Product item_0219",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 145.59,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0220",
    "title": "Product item_0220",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 49.91,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0221",
    "title": "Product item_0221",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 119.94,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0222",
    "title": "Product item_0222",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 164.45,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0223",
    "title": "Product item_0223",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 36.4,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0224",
    "title": "Product item_0224",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 233,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0225",
    "title": "Product item_0225",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 128.9,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0226",
    "title": "Product item_0226",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 205.78,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0227",
    "title": "Product item_0227",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 165.21,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0228",
    "title": "Product item_0228",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 178.95,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0229",
    "title": "Product item_0229",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 180.45,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0230",
    "title": "Product item_0230",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 219.4,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0231",
    "title": "Product item_0231",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 155.33,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0232",
    "title": "Product item_0232",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 73.16,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0233",
    "title": "Product item_0233",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 29.38,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0234",
    "title": "Product item_0234",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 153.53,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0235",
    "title": "Product item_0235",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 21.26,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0236",
    "title": "Product item_0236",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 64.12,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0237",
    "title": "Product item_0237",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 19.87,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0238",
    "title": "Product item_0238",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 139.48,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0239",
    "title": "Product item_0239",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 44.28,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0240",
    "title": "Product item_0240",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 154.55,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0241",
    "title": "Product item_0241",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 198.76,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0242",
    "title": "Product item_0242",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 206.36,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0243",
    "title": "Product item_0243",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 80.84,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0244",
    "title": "Product item_0244",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 139.49,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0245",
    "title": "Product item_0245",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 142.94,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0246",
    "title": "Product item_0246",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 120.86,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0247",
    "title": "Product item_0247",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 165.5,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0248",
    "title": "Product item_0248",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 56.47,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0249",
    "title": "Product item_0249",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 244.91,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0250",
    "title": "Product item_0250",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 29.03,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0251",
    "title": "Product item_0251",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 208.16,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0252",
    "title": "Product item_0252",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 150.55,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0253",
    "title": "Product item_0253",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 64.99,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0254",
    "title": "Product item_0254",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 187.69,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0255",
    "title": "Product item_0255",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 25.7,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0256",
    "title": "Product item_0256",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 194.73,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0257",
    "title": "Product item_0257",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 149.55,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0258",
    "title": "Product item_0258",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 32.13,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0259",
    "title": "Product item_0259",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 24.48,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0260",
    "title": "Product item_0260",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 188.95,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0261",
    "title": "Product item_0261",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 76.77,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0262",
    "title": "Product item_0262",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 222.88,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0263",
    "title": "Product item_0263",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 191.71,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0264",
    "title": "Product item_0264",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 176.06,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0265",
    "title": "Product item_0265",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 73.26,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0266",
    "title": "Product item_0266",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 178.69,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0267",
    "title": "Product item_0267",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 237.3,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0268",
    "title": "Product item_0268",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 27.43,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0269",
    "title": "Product item_0269",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 84.91,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0270",
    "title": "Product item_0270",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 62.49,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0271",
    "title": "Product item_0271",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 101.35,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0272",
    "title": "Product item_0272",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 186.87,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0273",
    "title": "Product item_0273",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 121.19,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0274",
    "title": "Product item_0274",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 144.83,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0275",
    "title": "Product item_0275",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 81.97,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0276",
    "title": "Product item_0276",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 39.62,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0277",
    "title": "Product item_0277",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 70.4,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0278",
    "title": "Product item_0278",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 215.62,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0279",
    "title": "Product item_0279",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 152.59,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0280",
    "title": "Product item_0280",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 97.61,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0281",
    "title": "Product item_0281",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 198.53,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0282",
    "title": "Product item_0282",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 20.79,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0283",
    "title": "Product item_0283",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 67.2,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0284",
    "title": "Product item_0284",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 61.38,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0285",
    "title": "Product item_0285",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 246.41,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0286",
    "title": "Product item_0286",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 177.99,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0287",
    "title": "Product item_0287",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 156.36,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0288",
    "title": "Product item_0288",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 6.22,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0289",
    "title": "Product item_0289",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 50.16,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0290",
    "title": "Product item_0290",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 160.71,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0291",
    "title": "Product item_0291",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 98.03,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0292",
    "title": "Product item_0292",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 231.45,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0293",
    "title": "Product item_0293",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 192.36,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0294",
    "title": "Product item_0294",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 114.61,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0295",
    "title": "Product item_0295",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 188.63,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0296",
    "title": "Product item_0296",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 243.97,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0297",
    "title": "Product item_0297",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 117.94,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0298",
    "title": "Product item_0298",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 162.42,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0299",
    "title": "Product item_0299",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 230.07,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0300",
    "title": "Product item_0300",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 65.93,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0301",
    "title": "Product item_0301",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 222.21,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0302",
    "title": "Product item_0302",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 28.9,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0303",
    "title": "Product item_0303",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 163.87,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0304",
    "title": "Product item_0304",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 196.8,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0305",
    "title": "Product item_0305",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 155.58,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0306",
    "title": "Product item_0306",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 126.41,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0307",
    "title": "Product item_0307",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 156.25,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0308",
    "title": "Product item_0308",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 42.16,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0309",
    "title": "Product item_0309",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 57.81,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0310",
    "title": "Product item_0310",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 71.13,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0311",
    "title": "Product item_0311",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 151.46,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0312",
    "title": "Product item_0312",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 113.27,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0313",
    "title": "Product item_0313",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 64.9,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0314",
    "title": "Product item_0314",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 221.29,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0315",
    "title": "Product item_0315",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 96.61,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0316",
    "title": "Product item_0316",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 114.28,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0317",
    "title": "Product item_0317",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 105.38,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0318",
    "title": "Product item_0318",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 49.58,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0319",
    "title": "Product item_0319",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 184.25,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0320",
    "title": "Product item_0320",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 114.87,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0321",
    "title": "Product item_0321",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 8.98,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0322",
    "title": "Product item_0322",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 241.18,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0323",
    "title": "Product item_0323",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 204.86,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0324",
    "title": "Product item_0324",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 182.27,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0325",
    "title": "Product item_0325",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 217.93,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0326",
    "title": "Product item_0326",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 227.17,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0327",
    "title": "Product item_0327",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 87.68,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0328",
    "title": "Product item_0328",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 154.68,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0329",
    "title": "Product item_0329",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 154.16,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0330",
    "title": "Product item_0330",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 70.49,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0331",
    "title": "Product item_0331",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 59.38,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0332",
    "title": "Product item_0332",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 32.91,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0333",
    "title": "Product item_0333",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 166.53,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0334",
    "title": "Product item_0334",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 196.33,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0335",
    "title": "Product item_0335",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 191.43,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0336",
    "title": "Product item_0336",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 82.49,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0337",
    "title": "Product item_0337",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 55.29,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0338",
    "title": "Product item_0338",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 116.14,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0339",
    "title": "Product item_0339",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 138.97,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0340",
    "title": "Product item_0340",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 230.18,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0341",
    "title": "Product item_0341",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 219.33,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0342",
    "title": "Product item_0342",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 229.43,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0343",
    "title": "Product item_0343",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 213.31,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0344",
    "title": "Product item_0344",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 241.32,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0345",
    "title": "Product item_0345",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 119.05,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0346",
    "title": "Product item_0346",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 16.63,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0347",
    "title": "Product item_0347",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 25.47,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0348",
    "title": "Product item_0348",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 100.62,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0349",
    "title": "Product item_0349",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 68.13,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0350",
    "title": "Product item_0350",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 46.64,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0351",
    "title": "Product item_0351",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 238.53,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0352",
    "title": "Product item_0352",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 52.88,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0353",
    "title": "Product item_0353",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 240.05,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0354",
    "title": "Product item_0354",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 191.41,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0355",
    "title": "Product item_0355",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 162.03,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0356",
    "title": "Product item_0356",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 170.62,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0357",
    "title": "Product item_0357",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 86.21,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0358",
    "title": "Product item_0358",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 163.37,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0359",
    "title": "Product item_0359",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 50.74,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0360",
    "title": "Product item_0360",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 234.74,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0361",
    "title": "Product item_0361",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 78.94,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0362",
    "title": "Product item_0362",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 67.52,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0363",
    "title": "Product item_0363",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 179.23,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0364",
    "title": "Product item_0364",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 165.83,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0365",
    "title": "Product item_0365",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 115.57,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0366",
    "title": "Product item_0366",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 159.95,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0367",
    "title": "Product item_0367",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 44.08,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0368",
    "title": "Product item_0368",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 169.97,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0369",
    "title": "Product item_0369",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 71.3,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0370",
    "title": "Product item_0370",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 71.84,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0371",
    "title": "Product item_0371",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 44.73,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0372",
    "title": "Product item_0372",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 86.43,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0373",
    "title": "Product item_0373",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 86.16,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0374",
    "title": "Product item_0374",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 116.27,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0375",
    "title": "Product item_0375",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 51.65,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0376",
    "title": "Product item_0376",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 52.29,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0377",
    "title": "Product item_0377",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 137.73,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0378",
    "title": "Product item_0378",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 36.57,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0379",
    "title": "Product item_0379",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 173.78,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0380",
    "title": "Product item_0380",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 236.27,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0381",
    "title": "Product item_0381",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 109.56,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0382",
    "title": "Product item_0382",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 225.77,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0383",
    "title": "Product item_0383",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 29.09,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0384",
    "title": "Product item_0384",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 15.52,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0385",
    "title": "Product item_0385",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 181.74,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0386",
    "title": "Product item_0386",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 191.45,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0387",
    "title": "Product item_0387",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 211.55,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0388",
    "title": "Product item_0388",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 18.92,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0389",
    "title": "Product item_0389",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 195.34,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0390",
    "title": "Product item_0390",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 97.25,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0391",
    "title": "Product item_0391",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 79.32,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0392",
    "title": "Product item_0392",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 47.39,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0393",
    "title": "Product item_0393",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 249.02,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0394",
    "title": "Product item_0394",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 20.6,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0395",
    "title": "Product item_0395",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 207.62,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0396",
    "title": "Product item_0396",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag3"
    ],
    "price": 161.68,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0397",
    "title": "Product item_0397",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 185.09,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0398",
    "title": "Product item_0398",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 179.37,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0399",
    "title": "Product item_0399",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 215.32,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0400",
    "title": "Product item_0400",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 50.03,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0401",
    "title": "Product item_0401",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 5.2,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0402",
    "title": "Product item_0402",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 151.64,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0403",
    "title": "Product item_0403",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 91.7,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0404",
    "title": "Product item_0404",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 35.33,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0405",
    "title": "Product item_0405",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 50.6,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0406",
    "title": "Product item_0406",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 86.28,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0407",
    "title": "Product item_0407",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 22.21,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0408",
    "title": "Product item_0408",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 18.94,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0409",
    "title": "Product item_0409",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 229.43,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0410",
    "title": "Product item_0410",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 81.47,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0411",
    "title": "Product item_0411",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 245.65,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0412",
    "title": "Product item_0412",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 241.27,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0413",
    "title": "Product item_0413",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 39.93,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0414",
    "title": "Product item_0414",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 180.16,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0415",
    "title": "Product item_0415",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 152.2,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0416",
    "title": "Product item_0416",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 174.21,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0417",
    "title": "Product item_0417",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 28.57,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0418",
    "title": "Product item_0418",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 51.2,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0419",
    "title": "Product item_0419",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 23.78,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0420",
    "title": "Product item_0420",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 153.01,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0421",
    "title": "Product item_0421",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 246.71,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0422",
    "title": "Product item_0422",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 169.64,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0423",
    "title": "Product item_0423",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 179.14,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0424",
    "title": "Product item_0424",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 246.22,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0425",
    "title": "Product item_0425",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 171.11,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0426",
    "title": "Product item_0426",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 14.84,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0427",
    "title": "Product item_0427",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 217.42,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0428",
    "title": "Product item_0428",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 94.35,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0429",
    "title": "Product item_0429",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 6.07,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0430",
    "title": "Product item_0430",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 129.27,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0431",
    "title": "Product item_0431",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 234.57,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0432",
    "title": "Product item_0432",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 68.05,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0433",
    "title": "Product item_0433",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 125.42,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0434",
    "title": "Product item_0434",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 47.58,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0435",
    "title": "Product item_0435",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 19.13,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0436",
    "title": "Product item_0436",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 24.73,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0437",
    "title": "Product item_0437",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 183.12,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0438",
    "title": "Product item_0438",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag3"
    ],
    "price": 216.7,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0439",
    "title": "Product item_0439",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 40.11,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0440",
    "title": "Product item_0440",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 75.64,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0441",
    "title": "Product item_0441",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 52.78,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0442",
    "title": "Product item_0442",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 11.58,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0443",
    "title": "Product item_0443",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 140.59,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0444",
    "title": "Product item_0444",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 176.48,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0445",
    "title": "Product item_0445",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 163.73,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0446",
    "title": "Product item_0446",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 100.07,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0447",
    "title": "Product item_0447",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 140.98,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0448",
    "title": "Product item_0448",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 28.05,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0449",
    "title": "Product item_0449",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 42.8,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0450",
    "title": "Product item_0450",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 233.29,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0451",
    "title": "Product item_0451",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 191.57,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0452",
    "title": "Product item_0452",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 231.07,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0453",
    "title": "Product item_0453",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 229.59,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0454",
    "title": "Product item_0454",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 164.1,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0455",
    "title": "Product item_0455",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 232.42,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0456",
    "title": "Product item_0456",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 107.09,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0457",
    "title": "Product item_0457",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 5.89,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0458",
    "title": "Product item_0458",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 20.33,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0459",
    "title": "Product item_0459",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 31.82,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0460",
    "title": "Product item_0460",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 28.36,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0461",
    "title": "Product item_0461",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 91.17,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0462",
    "title": "Product item_0462",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 176.81,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0463",
    "title": "Product item_0463",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 198.71,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0464",
    "title": "Product item_0464",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 149.6,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0465",
    "title": "Product item_0465",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 40.88,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0466",
    "title": "Product item_0466",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 188.24,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0467",
    "title": "Product item_0467",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 91.93,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0468",
    "title": "Product item_0468",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 8.18,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0469",
    "title": "Product item_0469",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 73.74,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0470",
    "title": "Product item_0470",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 86,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0471",
    "title": "Product item_0471",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 72.19,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0472",
    "title": "Product item_0472",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 203.91,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0473",
    "title": "Product item_0473",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 191.18,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0474",
    "title": "Product item_0474",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 175.41,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0475",
    "title": "Product item_0475",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 39.06,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0476",
    "title": "Product item_0476",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 202.92,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0477",
    "title": "Product item_0477",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 170.91,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0478",
    "title": "Product item_0478",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 52.74,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0479",
    "title": "Product item_0479",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 163.78,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0480",
    "title": "Product item_0480",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 123.46,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0481",
    "title": "Product item_0481",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 112.64,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0482",
    "title": "Product item_0482",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 32.47,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0483",
    "title": "Product item_0483",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 131.98,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0484",
    "title": "Product item_0484",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 29.48,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0485",
    "title": "Product item_0485",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 45.24,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0486",
    "title": "Product item_0486",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 37.47,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0487",
    "title": "Product item_0487",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 238.76,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0488",
    "title": "Product item_0488",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 74.79,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0489",
    "title": "Product item_0489",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 92.03,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0490",
    "title": "Product item_0490",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 83.91,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0491",
    "title": "Product item_0491",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 225.07,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0492",
    "title": "Product item_0492",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 142.93,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0493",
    "title": "Product item_0493",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 230.19,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0494",
    "title": "Product item_0494",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 78.48,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0495",
    "title": "Product item_0495",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 33.13,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0496",
    "title": "Product item_0496",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 141.27,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0497",
    "title": "Product item_0497",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 89.66,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0498",
    "title": "Product item_0498",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 201.47,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0499",
    "title": "Product item_0499",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 220.46,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0500",
    "title": "Product item_0500",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 120.78,
    "description": "A electronics product for testing recommendations."
  }
]
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
{
  "user_id": "user_00003",
  "segment": "vip",
  "notes": "Sample notes for user_00003 in segment vip."
}
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
[]
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
    "user_id": "user_001",
    "interactions": [
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
  },
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

generate **3** personalized recommendations for the user described in **Section 3** (`user_profile.user_id`).

**Constraints:**

- Exclude items already purchased or disliked by the Target User.
- Prefer items in categories the user engaged with recently.
- Balance exploitation with one exploratory item.

---

## 7. Expected LLM Response Format

The model **must** respond with valid JSON only (no markdown wrapper):

```json
{
  "user_id": "{user_id}",
  "generated_at": "2026-06-29T16:07:50Z",
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
| `generated_at` | ISO 8601 | `2026-06-29T16:07:50Z` — when recommendations were produced |
| `recommendations` | array | Ordered list, best first; length = `3` |
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
| `3` | integer | Number of recommendations to return |
| `2026-06-29T16:07:50Z` | ISO 8601 | Timestamp for the response (optional pre-fill) |
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
    "item_id": "item_0001",
    "title": "Product item_0001",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 60.2,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0002",
    "title": "Product item_0002",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 182.38,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0003",
    "title": "Product item_0003",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 210.16,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0004",
    "title": "Product item_0004",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 24.79,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0005",
    "title": "Product item_0005",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 221.11,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0006",
    "title": "Product item_0006",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 200.39,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0007",
    "title": "Product item_0007",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 218.07,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0008",
    "title": "Product item_0008",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 176.11,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0009",
    "title": "Product item_0009",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 170.17,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0010",
    "title": "Product item_0010",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 133.65,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0011",
    "title": "Product item_0011",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 164.93,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0012",
    "title": "Product item_0012",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 230.18,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0013",
    "title": "Product item_0013",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 29.63,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0014",
    "title": "Product item_0014",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 82.36,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0015",
    "title": "Product item_0015",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 22.75,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0016",
    "title": "Product item_0016",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 236.48,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0017",
    "title": "Product item_0017",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 114.09,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0018",
    "title": "Product item_0018",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 170.46,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0019",
    "title": "Product item_0019",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 211.07,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0020",
    "title": "Product item_0020",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 247.8,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0021",
    "title": "Product item_0021",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 44.64,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0022",
    "title": "Product item_0022",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 200.88,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0023",
    "title": "Product item_0023",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 222.42,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0024",
    "title": "Product item_0024",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 16.97,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0025",
    "title": "Product item_0025",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 189.75,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0026",
    "title": "Product item_0026",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 91.91,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0027",
    "title": "Product item_0027",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 163.91,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0028",
    "title": "Product item_0028",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 130.94,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0029",
    "title": "Product item_0029",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 68.64,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0030",
    "title": "Product item_0030",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 38.13,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0031",
    "title": "Product item_0031",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 33.65,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0032",
    "title": "Product item_0032",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 30.31,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0033",
    "title": "Product item_0033",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 161.26,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0034",
    "title": "Product item_0034",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 91.1,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0035",
    "title": "Product item_0035",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 58.08,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0036",
    "title": "Product item_0036",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 174.51,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0037",
    "title": "Product item_0037",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 173.28,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0038",
    "title": "Product item_0038",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 137.1,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0039",
    "title": "Product item_0039",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 234.06,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0040",
    "title": "Product item_0040",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 99.43,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0041",
    "title": "Product item_0041",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 209.79,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0042",
    "title": "Product item_0042",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 23.41,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0043",
    "title": "Product item_0043",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 186.26,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0044",
    "title": "Product item_0044",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 247.17,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0045",
    "title": "Product item_0045",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 212.42,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0046",
    "title": "Product item_0046",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 147.02,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0047",
    "title": "Product item_0047",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 158.56,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0048",
    "title": "Product item_0048",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 91.56,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0049",
    "title": "Product item_0049",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 171.47,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0050",
    "title": "Product item_0050",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 193.75,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0051",
    "title": "Product item_0051",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 249.57,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0052",
    "title": "Product item_0052",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 125.65,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0053",
    "title": "Product item_0053",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 20.02,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0054",
    "title": "Product item_0054",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 150.99,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0055",
    "title": "Product item_0055",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 34.16,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0056",
    "title": "Product item_0056",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 64.97,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0057",
    "title": "Product item_0057",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 85.55,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0058",
    "title": "Product item_0058",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 184.1,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0059",
    "title": "Product item_0059",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 50.03,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0060",
    "title": "Product item_0060",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 45.61,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0061",
    "title": "Product item_0061",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 16.02,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0062",
    "title": "Product item_0062",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 218.94,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0063",
    "title": "Product item_0063",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 174.91,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0064",
    "title": "Product item_0064",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 14.79,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0065",
    "title": "Product item_0065",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 11.93,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0066",
    "title": "Product item_0066",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 141.88,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0067",
    "title": "Product item_0067",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 74.22,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0068",
    "title": "Product item_0068",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 151.2,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0069",
    "title": "Product item_0069",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 111.48,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0070",
    "title": "Product item_0070",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 231.21,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0071",
    "title": "Product item_0071",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 75.51,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0072",
    "title": "Product item_0072",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 171.39,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0073",
    "title": "Product item_0073",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 18.24,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0074",
    "title": "Product item_0074",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 239.19,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0075",
    "title": "Product item_0075",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 245.29,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0076",
    "title": "Product item_0076",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 186.07,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0077",
    "title": "Product item_0077",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 118.98,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0078",
    "title": "Product item_0078",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 58.9,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0079",
    "title": "Product item_0079",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 204.07,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0080",
    "title": "Product item_0080",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 185.41,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0081",
    "title": "Product item_0081",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 59.38,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0082",
    "title": "Product item_0082",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 115.55,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0083",
    "title": "Product item_0083",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 118.39,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0084",
    "title": "Product item_0084",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 207.47,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0085",
    "title": "Product item_0085",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 172.54,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0086",
    "title": "Product item_0086",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 162.58,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0087",
    "title": "Product item_0087",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 164.21,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0088",
    "title": "Product item_0088",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 71.27,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0089",
    "title": "Product item_0089",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 96.07,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0090",
    "title": "Product item_0090",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 16.19,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0091",
    "title": "Product item_0091",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 197.8,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0092",
    "title": "Product item_0092",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 111.08,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0093",
    "title": "Product item_0093",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 112.37,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0094",
    "title": "Product item_0094",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 65.11,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0095",
    "title": "Product item_0095",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 56.31,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0096",
    "title": "Product item_0096",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 149.02,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0097",
    "title": "Product item_0097",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 61.98,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0098",
    "title": "Product item_0098",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 20.73,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0099",
    "title": "Product item_0099",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 193.04,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0100",
    "title": "Product item_0100",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 97.24,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0101",
    "title": "Product item_0101",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 98.16,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0102",
    "title": "Product item_0102",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 110.67,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0103",
    "title": "Product item_0103",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 204.69,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0104",
    "title": "Product item_0104",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 83.61,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0105",
    "title": "Product item_0105",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 190.72,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0106",
    "title": "Product item_0106",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 235.19,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0107",
    "title": "Product item_0107",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 166.91,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0108",
    "title": "Product item_0108",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 110.76,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0109",
    "title": "Product item_0109",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 188.21,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0110",
    "title": "Product item_0110",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 117.18,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0111",
    "title": "Product item_0111",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 106.53,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0112",
    "title": "Product item_0112",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 133.69,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0113",
    "title": "Product item_0113",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 82.84,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0114",
    "title": "Product item_0114",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 30.54,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0115",
    "title": "Product item_0115",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 106.64,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0116",
    "title": "Product item_0116",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 35.34,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0117",
    "title": "Product item_0117",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 105.56,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0118",
    "title": "Product item_0118",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 141.45,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0119",
    "title": "Product item_0119",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 233.02,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0120",
    "title": "Product item_0120",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 13.87,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0121",
    "title": "Product item_0121",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 157.26,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0122",
    "title": "Product item_0122",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 117.15,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0123",
    "title": "Product item_0123",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 9.96,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0124",
    "title": "Product item_0124",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 66.14,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0125",
    "title": "Product item_0125",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 34.03,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0126",
    "title": "Product item_0126",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 58.11,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0127",
    "title": "Product item_0127",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 72.27,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0128",
    "title": "Product item_0128",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 103.32,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0129",
    "title": "Product item_0129",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 134.61,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0130",
    "title": "Product item_0130",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 5.24,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0131",
    "title": "Product item_0131",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 44.66,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0132",
    "title": "Product item_0132",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 27.92,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0133",
    "title": "Product item_0133",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 139.67,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0134",
    "title": "Product item_0134",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 85.62,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0135",
    "title": "Product item_0135",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 139.36,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0136",
    "title": "Product item_0136",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 59.35,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0137",
    "title": "Product item_0137",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 10.94,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0138",
    "title": "Product item_0138",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 235.63,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0139",
    "title": "Product item_0139",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 31.54,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0140",
    "title": "Product item_0140",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 36.08,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0141",
    "title": "Product item_0141",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 40.9,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0142",
    "title": "Product item_0142",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 245.41,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0143",
    "title": "Product item_0143",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 156.33,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0144",
    "title": "Product item_0144",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 154.52,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0145",
    "title": "Product item_0145",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 246.74,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0146",
    "title": "Product item_0146",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 26.67,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0147",
    "title": "Product item_0147",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 243.75,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0148",
    "title": "Product item_0148",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 54.98,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0149",
    "title": "Product item_0149",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 157.65,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0150",
    "title": "Product item_0150",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 107.92,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0151",
    "title": "Product item_0151",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag3"
    ],
    "price": 99.07,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0152",
    "title": "Product item_0152",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 233.15,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0153",
    "title": "Product item_0153",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 41.56,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0154",
    "title": "Product item_0154",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 249.28,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0155",
    "title": "Product item_0155",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 97.89,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0156",
    "title": "Product item_0156",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 163.95,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0157",
    "title": "Product item_0157",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 242.14,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0158",
    "title": "Product item_0158",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag3"
    ],
    "price": 19.11,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0159",
    "title": "Product item_0159",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 231.1,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0160",
    "title": "Product item_0160",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 144.11,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0161",
    "title": "Product item_0161",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 177.58,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0162",
    "title": "Product item_0162",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 41.55,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0163",
    "title": "Product item_0163",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 135.68,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0164",
    "title": "Product item_0164",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 170.01,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0165",
    "title": "Product item_0165",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 116.78,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0166",
    "title": "Product item_0166",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 42.97,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0167",
    "title": "Product item_0167",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 64.81,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0168",
    "title": "Product item_0168",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 214.33,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0169",
    "title": "Product item_0169",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 13.45,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0170",
    "title": "Product item_0170",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 178.67,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0171",
    "title": "Product item_0171",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 249.85,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0172",
    "title": "Product item_0172",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 6.83,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0173",
    "title": "Product item_0173",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 44.46,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0174",
    "title": "Product item_0174",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 170.17,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0175",
    "title": "Product item_0175",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 19.7,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0176",
    "title": "Product item_0176",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 149.5,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0177",
    "title": "Product item_0177",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 172.58,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0178",
    "title": "Product item_0178",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 149.1,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0179",
    "title": "Product item_0179",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag3"
    ],
    "price": 94.47,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0180",
    "title": "Product item_0180",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 175.83,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0181",
    "title": "Product item_0181",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 150.6,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0182",
    "title": "Product item_0182",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 55.87,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0183",
    "title": "Product item_0183",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 142.97,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0184",
    "title": "Product item_0184",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 221.07,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0185",
    "title": "Product item_0185",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 168.37,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0186",
    "title": "Product item_0186",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 120.59,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0187",
    "title": "Product item_0187",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 83.79,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0188",
    "title": "Product item_0188",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 26.04,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0189",
    "title": "Product item_0189",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 64.93,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0190",
    "title": "Product item_0190",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 7.74,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0191",
    "title": "Product item_0191",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 246.72,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0192",
    "title": "Product item_0192",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 59.51,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0193",
    "title": "Product item_0193",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 204.64,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0194",
    "title": "Product item_0194",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 46.32,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0195",
    "title": "Product item_0195",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 74.48,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0196",
    "title": "Product item_0196",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 107.48,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0197",
    "title": "Product item_0197",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 84.48,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0198",
    "title": "Product item_0198",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 200.6,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0199",
    "title": "Product item_0199",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 239.63,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0200",
    "title": "Product item_0200",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 219.45,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0201",
    "title": "Product item_0201",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 42.63,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0202",
    "title": "Product item_0202",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 195.29,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0203",
    "title": "Product item_0203",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 41.25,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0204",
    "title": "Product item_0204",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 228.78,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0205",
    "title": "Product item_0205",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 131.88,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0206",
    "title": "Product item_0206",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 87.41,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0207",
    "title": "Product item_0207",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 43.47,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0208",
    "title": "Product item_0208",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 7.89,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0209",
    "title": "Product item_0209",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 65.75,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0210",
    "title": "Product item_0210",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 228.93,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0211",
    "title": "Product item_0211",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 200.33,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0212",
    "title": "Product item_0212",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 242.6,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0213",
    "title": "Product item_0213",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 29.69,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0214",
    "title": "Product item_0214",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 150.08,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0215",
    "title": "Product item_0215",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 68,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0216",
    "title": "Product item_0216",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 138.02,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0217",
    "title": "Product item_0217",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 83.51,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0218",
    "title": "Product item_0218",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 74.45,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0219",
    "title": "Product item_0219",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 145.59,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0220",
    "title": "Product item_0220",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 49.91,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0221",
    "title": "Product item_0221",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 119.94,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0222",
    "title": "Product item_0222",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 164.45,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0223",
    "title": "Product item_0223",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 36.4,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0224",
    "title": "Product item_0224",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 233,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0225",
    "title": "Product item_0225",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 128.9,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0226",
    "title": "Product item_0226",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 205.78,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0227",
    "title": "Product item_0227",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 165.21,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0228",
    "title": "Product item_0228",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 178.95,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0229",
    "title": "Product item_0229",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 180.45,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0230",
    "title": "Product item_0230",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 219.4,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0231",
    "title": "Product item_0231",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 155.33,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0232",
    "title": "Product item_0232",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 73.16,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0233",
    "title": "Product item_0233",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 29.38,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0234",
    "title": "Product item_0234",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 153.53,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0235",
    "title": "Product item_0235",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 21.26,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0236",
    "title": "Product item_0236",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 64.12,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0237",
    "title": "Product item_0237",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 19.87,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0238",
    "title": "Product item_0238",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 139.48,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0239",
    "title": "Product item_0239",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 44.28,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0240",
    "title": "Product item_0240",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 154.55,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0241",
    "title": "Product item_0241",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 198.76,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0242",
    "title": "Product item_0242",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 206.36,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0243",
    "title": "Product item_0243",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 80.84,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0244",
    "title": "Product item_0244",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 139.49,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0245",
    "title": "Product item_0245",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 142.94,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0246",
    "title": "Product item_0246",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 120.86,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0247",
    "title": "Product item_0247",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 165.5,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0248",
    "title": "Product item_0248",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 56.47,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0249",
    "title": "Product item_0249",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 244.91,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0250",
    "title": "Product item_0250",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 29.03,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0251",
    "title": "Product item_0251",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 208.16,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0252",
    "title": "Product item_0252",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 150.55,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0253",
    "title": "Product item_0253",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 64.99,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0254",
    "title": "Product item_0254",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 187.69,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0255",
    "title": "Product item_0255",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 25.7,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0256",
    "title": "Product item_0256",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 194.73,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0257",
    "title": "Product item_0257",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 149.55,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0258",
    "title": "Product item_0258",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 32.13,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0259",
    "title": "Product item_0259",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 24.48,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0260",
    "title": "Product item_0260",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 188.95,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0261",
    "title": "Product item_0261",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 76.77,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0262",
    "title": "Product item_0262",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 222.88,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0263",
    "title": "Product item_0263",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 191.71,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0264",
    "title": "Product item_0264",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 176.06,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0265",
    "title": "Product item_0265",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 73.26,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0266",
    "title": "Product item_0266",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 178.69,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0267",
    "title": "Product item_0267",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 237.3,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0268",
    "title": "Product item_0268",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 27.43,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0269",
    "title": "Product item_0269",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 84.91,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0270",
    "title": "Product item_0270",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 62.49,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0271",
    "title": "Product item_0271",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 101.35,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0272",
    "title": "Product item_0272",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 186.87,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0273",
    "title": "Product item_0273",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 121.19,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0274",
    "title": "Product item_0274",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 144.83,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0275",
    "title": "Product item_0275",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 81.97,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0276",
    "title": "Product item_0276",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 39.62,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0277",
    "title": "Product item_0277",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 70.4,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0278",
    "title": "Product item_0278",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 215.62,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0279",
    "title": "Product item_0279",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 152.59,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0280",
    "title": "Product item_0280",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 97.61,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0281",
    "title": "Product item_0281",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 198.53,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0282",
    "title": "Product item_0282",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 20.79,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0283",
    "title": "Product item_0283",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 67.2,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0284",
    "title": "Product item_0284",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 61.38,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0285",
    "title": "Product item_0285",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 246.41,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0286",
    "title": "Product item_0286",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 177.99,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0287",
    "title": "Product item_0287",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 156.36,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0288",
    "title": "Product item_0288",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 6.22,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0289",
    "title": "Product item_0289",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 50.16,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0290",
    "title": "Product item_0290",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 160.71,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0291",
    "title": "Product item_0291",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 98.03,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0292",
    "title": "Product item_0292",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 231.45,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0293",
    "title": "Product item_0293",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 192.36,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0294",
    "title": "Product item_0294",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 114.61,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0295",
    "title": "Product item_0295",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 188.63,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0296",
    "title": "Product item_0296",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 243.97,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0297",
    "title": "Product item_0297",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 117.94,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0298",
    "title": "Product item_0298",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 162.42,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0299",
    "title": "Product item_0299",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 230.07,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0300",
    "title": "Product item_0300",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 65.93,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0301",
    "title": "Product item_0301",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 222.21,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0302",
    "title": "Product item_0302",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 28.9,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0303",
    "title": "Product item_0303",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 163.87,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0304",
    "title": "Product item_0304",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 196.8,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0305",
    "title": "Product item_0305",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 155.58,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0306",
    "title": "Product item_0306",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 126.41,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0307",
    "title": "Product item_0307",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 156.25,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0308",
    "title": "Product item_0308",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 42.16,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0309",
    "title": "Product item_0309",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 57.81,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0310",
    "title": "Product item_0310",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 71.13,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0311",
    "title": "Product item_0311",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 151.46,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0312",
    "title": "Product item_0312",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 113.27,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0313",
    "title": "Product item_0313",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 64.9,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0314",
    "title": "Product item_0314",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 221.29,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0315",
    "title": "Product item_0315",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 96.61,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0316",
    "title": "Product item_0316",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 114.28,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0317",
    "title": "Product item_0317",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 105.38,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0318",
    "title": "Product item_0318",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 49.58,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0319",
    "title": "Product item_0319",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 184.25,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0320",
    "title": "Product item_0320",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 114.87,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0321",
    "title": "Product item_0321",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 8.98,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0322",
    "title": "Product item_0322",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 241.18,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0323",
    "title": "Product item_0323",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 204.86,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0324",
    "title": "Product item_0324",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 182.27,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0325",
    "title": "Product item_0325",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 217.93,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0326",
    "title": "Product item_0326",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 227.17,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0327",
    "title": "Product item_0327",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 87.68,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0328",
    "title": "Product item_0328",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 154.68,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0329",
    "title": "Product item_0329",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 154.16,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0330",
    "title": "Product item_0330",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 70.49,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0331",
    "title": "Product item_0331",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 59.38,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0332",
    "title": "Product item_0332",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 32.91,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0333",
    "title": "Product item_0333",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 166.53,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0334",
    "title": "Product item_0334",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 196.33,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0335",
    "title": "Product item_0335",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 191.43,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0336",
    "title": "Product item_0336",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 82.49,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0337",
    "title": "Product item_0337",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 55.29,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0338",
    "title": "Product item_0338",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 116.14,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0339",
    "title": "Product item_0339",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 138.97,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0340",
    "title": "Product item_0340",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 230.18,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0341",
    "title": "Product item_0341",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 219.33,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0342",
    "title": "Product item_0342",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 229.43,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0343",
    "title": "Product item_0343",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 213.31,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0344",
    "title": "Product item_0344",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 241.32,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0345",
    "title": "Product item_0345",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 119.05,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0346",
    "title": "Product item_0346",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 16.63,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0347",
    "title": "Product item_0347",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 25.47,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0348",
    "title": "Product item_0348",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 100.62,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0349",
    "title": "Product item_0349",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 68.13,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0350",
    "title": "Product item_0350",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 46.64,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0351",
    "title": "Product item_0351",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 238.53,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0352",
    "title": "Product item_0352",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 52.88,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0353",
    "title": "Product item_0353",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 240.05,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0354",
    "title": "Product item_0354",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 191.41,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0355",
    "title": "Product item_0355",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 162.03,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0356",
    "title": "Product item_0356",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 170.62,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0357",
    "title": "Product item_0357",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 86.21,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0358",
    "title": "Product item_0358",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 163.37,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0359",
    "title": "Product item_0359",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 50.74,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0360",
    "title": "Product item_0360",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 234.74,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0361",
    "title": "Product item_0361",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 78.94,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0362",
    "title": "Product item_0362",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 67.52,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0363",
    "title": "Product item_0363",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 179.23,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0364",
    "title": "Product item_0364",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 165.83,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0365",
    "title": "Product item_0365",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 115.57,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0366",
    "title": "Product item_0366",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 159.95,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0367",
    "title": "Product item_0367",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 44.08,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0368",
    "title": "Product item_0368",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 169.97,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0369",
    "title": "Product item_0369",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 71.3,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0370",
    "title": "Product item_0370",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 71.84,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0371",
    "title": "Product item_0371",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 44.73,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0372",
    "title": "Product item_0372",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 86.43,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0373",
    "title": "Product item_0373",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag1"
    ],
    "price": 86.16,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0374",
    "title": "Product item_0374",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 116.27,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0375",
    "title": "Product item_0375",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 51.65,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0376",
    "title": "Product item_0376",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 52.29,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0377",
    "title": "Product item_0377",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 137.73,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0378",
    "title": "Product item_0378",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 36.57,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0379",
    "title": "Product item_0379",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 173.78,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0380",
    "title": "Product item_0380",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 236.27,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0381",
    "title": "Product item_0381",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 109.56,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0382",
    "title": "Product item_0382",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 225.77,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0383",
    "title": "Product item_0383",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 29.09,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0384",
    "title": "Product item_0384",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 15.52,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0385",
    "title": "Product item_0385",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 181.74,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0386",
    "title": "Product item_0386",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 191.45,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0387",
    "title": "Product item_0387",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 211.55,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0388",
    "title": "Product item_0388",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 18.92,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0389",
    "title": "Product item_0389",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 195.34,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0390",
    "title": "Product item_0390",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 97.25,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0391",
    "title": "Product item_0391",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 79.32,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0392",
    "title": "Product item_0392",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 47.39,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0393",
    "title": "Product item_0393",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 249.02,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0394",
    "title": "Product item_0394",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 20.6,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0395",
    "title": "Product item_0395",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 207.62,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0396",
    "title": "Product item_0396",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag3"
    ],
    "price": 161.68,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0397",
    "title": "Product item_0397",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 185.09,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0398",
    "title": "Product item_0398",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 179.37,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0399",
    "title": "Product item_0399",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag6"
    ],
    "price": 215.32,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0400",
    "title": "Product item_0400",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 50.03,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0401",
    "title": "Product item_0401",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 5.2,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0402",
    "title": "Product item_0402",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 151.64,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0403",
    "title": "Product item_0403",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 91.7,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0404",
    "title": "Product item_0404",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag4"
    ],
    "price": 35.33,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0405",
    "title": "Product item_0405",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 50.6,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0406",
    "title": "Product item_0406",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 86.28,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0407",
    "title": "Product item_0407",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 22.21,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0408",
    "title": "Product item_0408",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 18.94,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0409",
    "title": "Product item_0409",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag2"
    ],
    "price": 229.43,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0410",
    "title": "Product item_0410",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 81.47,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0411",
    "title": "Product item_0411",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 245.65,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0412",
    "title": "Product item_0412",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 241.27,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0413",
    "title": "Product item_0413",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 39.93,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0414",
    "title": "Product item_0414",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 180.16,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0415",
    "title": "Product item_0415",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 152.2,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0416",
    "title": "Product item_0416",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 174.21,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0417",
    "title": "Product item_0417",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag3"
    ],
    "price": 28.57,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0418",
    "title": "Product item_0418",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 51.2,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0419",
    "title": "Product item_0419",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 23.78,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0420",
    "title": "Product item_0420",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 153.01,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0421",
    "title": "Product item_0421",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 246.71,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0422",
    "title": "Product item_0422",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 169.64,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0423",
    "title": "Product item_0423",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 179.14,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0424",
    "title": "Product item_0424",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 246.22,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0425",
    "title": "Product item_0425",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 171.11,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0426",
    "title": "Product item_0426",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 14.84,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0427",
    "title": "Product item_0427",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 217.42,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0428",
    "title": "Product item_0428",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 94.35,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0429",
    "title": "Product item_0429",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 6.07,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0430",
    "title": "Product item_0430",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag2"
    ],
    "price": 129.27,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0431",
    "title": "Product item_0431",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag3"
    ],
    "price": 234.57,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0432",
    "title": "Product item_0432",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 68.05,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0433",
    "title": "Product item_0433",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 125.42,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0434",
    "title": "Product item_0434",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 47.58,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0435",
    "title": "Product item_0435",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 19.13,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0436",
    "title": "Product item_0436",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 24.73,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0437",
    "title": "Product item_0437",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 183.12,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0438",
    "title": "Product item_0438",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag3"
    ],
    "price": 216.7,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0439",
    "title": "Product item_0439",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 40.11,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0440",
    "title": "Product item_0440",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag5"
    ],
    "price": 75.64,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0441",
    "title": "Product item_0441",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 52.78,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0442",
    "title": "Product item_0442",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag0"
    ],
    "price": 11.58,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0443",
    "title": "Product item_0443",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 140.59,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0444",
    "title": "Product item_0444",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag2"
    ],
    "price": 176.48,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0445",
    "title": "Product item_0445",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 163.73,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0446",
    "title": "Product item_0446",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag4"
    ],
    "price": 100.07,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0447",
    "title": "Product item_0447",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 140.98,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0448",
    "title": "Product item_0448",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 28.05,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0449",
    "title": "Product item_0449",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 42.8,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0450",
    "title": "Product item_0450",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 233.29,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0451",
    "title": "Product item_0451",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 191.57,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0452",
    "title": "Product item_0452",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 231.07,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0453",
    "title": "Product item_0453",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 229.59,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0454",
    "title": "Product item_0454",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag5"
    ],
    "price": 164.1,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0455",
    "title": "Product item_0455",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 232.42,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0456",
    "title": "Product item_0456",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag0"
    ],
    "price": 107.09,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0457",
    "title": "Product item_0457",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag1"
    ],
    "price": 5.89,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0458",
    "title": "Product item_0458",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 20.33,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0459",
    "title": "Product item_0459",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 31.82,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0460",
    "title": "Product item_0460",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 28.36,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0461",
    "title": "Product item_0461",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 91.17,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0462",
    "title": "Product item_0462",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag6"
    ],
    "price": 176.81,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0463",
    "title": "Product item_0463",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag0"
    ],
    "price": 198.71,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0464",
    "title": "Product item_0464",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag1"
    ],
    "price": 149.6,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0465",
    "title": "Product item_0465",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 40.88,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0466",
    "title": "Product item_0466",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 188.24,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0467",
    "title": "Product item_0467",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag4"
    ],
    "price": 91.93,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0468",
    "title": "Product item_0468",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag5"
    ],
    "price": 8.18,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0469",
    "title": "Product item_0469",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 73.74,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0470",
    "title": "Product item_0470",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 86,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0471",
    "title": "Product item_0471",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 72.19,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0472",
    "title": "Product item_0472",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag2"
    ],
    "price": 203.91,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0473",
    "title": "Product item_0473",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag3"
    ],
    "price": 191.18,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0474",
    "title": "Product item_0474",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag4"
    ],
    "price": 175.41,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0475",
    "title": "Product item_0475",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 39.06,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0476",
    "title": "Product item_0476",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag6"
    ],
    "price": 202.92,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0477",
    "title": "Product item_0477",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag0"
    ],
    "price": 170.91,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0478",
    "title": "Product item_0478",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 52.74,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0479",
    "title": "Product item_0479",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 163.78,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0480",
    "title": "Product item_0480",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag3"
    ],
    "price": 123.46,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0481",
    "title": "Product item_0481",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag4"
    ],
    "price": 112.64,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0482",
    "title": "Product item_0482",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag5"
    ],
    "price": 32.47,
    "description": "A electronics product for testing recommendations."
  },
  {
    "item_id": "item_0483",
    "title": "Product item_0483",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag6"
    ],
    "price": 131.98,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0484",
    "title": "Product item_0484",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag0"
    ],
    "price": 29.48,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0485",
    "title": "Product item_0485",
    "category": "home",
    "tags": [
      "home",
      "sample",
      "tag1"
    ],
    "price": 45.24,
    "description": "A home product for testing recommendations."
  },
  {
    "item_id": "item_0486",
    "title": "Product item_0486",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 37.47,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0487",
    "title": "Product item_0487",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 238.76,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0488",
    "title": "Product item_0488",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 74.79,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0489",
    "title": "Product item_0489",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag5"
    ],
    "price": 92.03,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0490",
    "title": "Product item_0490",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag6"
    ],
    "price": 83.91,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0491",
    "title": "Product item_0491",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 225.07,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0492",
    "title": "Product item_0492",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag1"
    ],
    "price": 142.93,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0493",
    "title": "Product item_0493",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag2"
    ],
    "price": 230.19,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0494",
    "title": "Product item_0494",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag3"
    ],
    "price": 78.48,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0495",
    "title": "Product item_0495",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag4"
    ],
    "price": 33.13,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0496",
    "title": "Product item_0496",
    "category": "books",
    "tags": [
      "books",
      "sample",
      "tag5"
    ],
    "price": 141.27,
    "description": "A books product for testing recommendations."
  },
  {
    "item_id": "item_0497",
    "title": "Product item_0497",
    "category": "sports",
    "tags": [
      "sports",
      "sample",
      "tag6"
    ],
    "price": 89.66,
    "description": "A sports product for testing recommendations."
  },
  {
    "item_id": "item_0498",
    "title": "Product item_0498",
    "category": "travel",
    "tags": [
      "travel",
      "sample",
      "tag0"
    ],
    "price": 201.47,
    "description": "A travel product for testing recommendations."
  },
  {
    "item_id": "item_0499",
    "title": "Product item_0499",
    "category": "fashion",
    "tags": [
      "fashion",
      "sample",
      "tag1"
    ],
    "price": 220.46,
    "description": "A fashion product for testing recommendations."
  },
  {
    "item_id": "item_0500",
    "title": "Product item_0500",
    "category": "electronics",
    "tags": [
      "electronics",
      "sample",
      "tag2"
    ],
    "price": 120.78,
    "description": "A electronics product for testing recommendations."
  }
]` | JSON array | Full item catalog (Section 2) |
| `{
  "user_id": "user_00003",
  "segment": "vip",
  "notes": "Sample notes for user_00003 in segment vip."
}` | JSON object | Target user profile (Section 3) |
| `[]` | JSON array | Target user event history (Section 4) |
| `[
  {
    "user_id": "user_001",
    "interactions": [
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
  },
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
| `# LLM-Based Recommendation Engine Generation Prompt

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
- **Catalog:** 5 items across 3 categories

### Algorithm Execution:

| Item | Category | Content Score | Collab Score | Final Score | Rank | Include? |
|------|----------|---|---|---|---|---|
| item_001 | electronics | 0.85 | 0.80 | 0.825 | - | ❌ Purchased |
| item_002 | books | 0.88 | 0.75 | 0.825 | - | ⚠️ Already liked |
| item_003 | books | 0.89 | 0.92 | 0.905 | **1** | ✅ **Recommend** |
| item_004 | electronics | 0.82 | 0.85 | 0.835 | **2** | ✅ **Recommend** |
| item_005 | travel | 0.68 | 0.55 | 0.625 | **3** | ✅ **Recommend** (Exploration) |

**Final Recommendations:** item_003 > item_004 > item_005

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

**Your recommendation engine should accept `user_id` and return personalized, explainable recommendations using the datasets above.**` | markdown | Additional instructions appended at the end |

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
- **Catalog:** 5 items across 3 categories

### Algorithm Execution:

| Item | Category | Content Score | Collab Score | Final Score | Rank | Include? |
|------|----------|---|---|---|---|---|
| item_001 | electronics | 0.85 | 0.80 | 0.825 | - | ❌ Purchased |
| item_002 | books | 0.88 | 0.75 | 0.825 | - | ⚠️ Already liked |
| item_003 | books | 0.89 | 0.92 | 0.905 | **1** | ✅ **Recommend** |
| item_004 | electronics | 0.82 | 0.85 | 0.835 | **2** | ✅ **Recommend** |
| item_005 | travel | 0.68 | 0.55 | 0.625 | **3** | ✅ **Recommend** (Exploration) |

**Final Recommendations:** item_003 > item_004 > item_005

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
