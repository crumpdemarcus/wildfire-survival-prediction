# Documentation

This folder contains all of the project's formal documentation — including the team's stakeholder presentation, a Q&A reference guide, and a final reflection report. These documents were created as part of the **ITAI 2377 Data Science in Artificial Intelligence** course at Houston City College (Spring 2026) for the WiDS Global Datathon 2026.

**Team:** DeMarcus Crump · Chloe Tu · Akinbobola Akinpelu · Aima Ayaz

---

## Folder Structure

```text
docs/
├── Final_Reflection_Report.pdf
├── Interview_Presentation.pdf
├── Project_Progress_Screenshots.html    ← HTML version of the screenshots doc
├── Project_Progress_Screenshots.md      ← Markdown version of the screenshots doc
└── Team_Presentation_Guidebook.pdf
```

---

## File Details

### `Interview_Presentation.pdf`
A stakeholder-facing slide deck used to communicate the project findings in an interview or business context. Covers:
- The problem background — why wildfire survival prediction matters for communities
- The dataset and competition context (WiDS Datathon 2026 on Kaggle)
- Our modeling strategy: why Survival Analysis was chosen over classification
- Physics-informed feature engineering (Danger Index, Trajectory Alignment, etc.)
- Model architecture: the ensemble of Random Survival Forest + Gradient Boosting Survival Analysis
- Validation methodology and results (0.9737 C-index)
- Actionable takeaways and real-world implications for emergency management

---

### `Team_Presentation_Guidebook.pdf`
A reference guide prepared to support Q&A sessions during presentations. Includes:
- A glossary of all technical terms used in the project (e.g., C-index, Survival Analysis, Kaplan-Meier, Brier Score)
- A plain-language "race car analogy" that explains hyperparameters as tunable dials
- Deep dives into the physics-based features and why each was engineered
- Explanations of why standard deep learning and gradient boosters were ruled out for this small dataset

---

### `Final_Reflection_Report.pdf`
A comprehensive team retrospective summarizing the full project experience. Includes:
- What each team member learned about data science and machine learning
- Challenges encountered during the competition (e.g., small dataset, class imbalance, right-censored survival data)
- How the project boosted each member's confidence in professional data science
- Long-term takeaways about the role of survival analysis in real-world domains
- Reflections on teamwork, workflow, and process

---

### `Project_Progress_Screenshots.md` / `.html`
A structured document created as part of the course submission requirements. It contains annotated screenshots documenting the full project journey, including:
- WiDS Datathon registration proofs
- Kaggle competition setup and data exploration
- Public leaderboard submission history
- Final private leaderboard score

The `.md` file is the primary version. The `.html` file is an alternative format for printing or sharing.
