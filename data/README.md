# CoCoLoFa Dataset

CoCoLoFa is a dataset of news comments containing common logical fallacies, designed to help models identify flawed reasoning in online discussions.

## Summary

- **Size:** 7,706 comments across 648 news articles.
- **Fallacies:** 8 types (Authority, Majority, Nature, Tradition, Worse Problems, False Dilemma, Hasty Generalization, Slippery Slope).
- **Labels:** Each comment is labeled as "none" or a specific fallacy type.

## Statistics

| Split     |  News   | Comments  |  Fallacy  |  Neutral  |
| :-------- | :-----: | :-------: | :-------: | :-------: |
| Train     |   452   |   5,370   |   3,168   |   2,202   |
| Dev       |   129   |   1,538   |    927    |    611    |
| Test      |   67    |    798    |    481    |    317    |
| **Total** | **648** | **7,706** | **4,576** | **3,130** |

## Data Format

Each entry in the JSON files follows this structure:

```json
{
  "id": 427,
  "title": "News Title",
  "content": "Full article text...",
  "comments": [
    {
      "id": "6078",
      "fallacy": "slippery slope",
      "comment": "Comment text...",
      "respond_to": ""
    }
  ]
}
```

### Key Fields

- `fallacy`: The fallacy type or "none".
- `comment`: The actual comment text.
- `respond_to`: ID of the comment being replied to (empty if it's a top-level comment).

## Citation

If you use this dataset, please cite:

```bibtex
@misc{yeh2024cocolofa,
      title={CoCoLoFa: A Dataset of News Comments with Common Logical Fallacies Written by LLM-Assisted Crowds},
      author={Min-Hsuan Yeh and Ruyuan Wan and Ting-Hao 'Kenneth' Huang},
      year={2024},
      url={https://arxiv.org/abs/2410.03457}
}
```
