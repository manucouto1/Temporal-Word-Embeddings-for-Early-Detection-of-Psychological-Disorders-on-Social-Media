# Temporal Word Embeddings for Early Detection of Psychological Disorders on Social Media

This project implements and evaluates a model based on **Temporal Word Embeddings (TWEC)** for early detection of psychological disorders from social media data. We compare the original TWEC model with an improved implementation integrated with LabChain.

---

## 📄 Abstract

Early detection of mental health disorders such as depression, anorexia, self-harm, and gambling addiction is crucial for timely intervention and support. This work explores the use of **temporal word embeddings** to capture semantic changes over time in users' social media posts and uses these embeddings to classify users according to their risk of developing these disorders.

We provide a comprehensive comparison between the original TWEC_SVM model and a LabChain-optimized version, evaluating performance across multiple longitudinal datasets.

---

## 📚 Methodology

- Publicly available social media datasets with confirmed cases of various psychological disorders are used, including depression, anorexia, self-harm, and gambling addiction.
- Temporal word embeddings are extracted to capture the semantic evolution of words over time.
- An SVM classifier is trained using these embeddings for early risk detection.
- The LabChain version integrates the model into an efficient pipeline, improving performance and data handling.

---

## 📊 Results

The following table presents the comparative performance of the original and LabChain-enhanced models across different datasets. Metrics include ERDE (Early Risk Detection Error) at 5 and 50 message windows, and the F1 score for balanced precision and recall.

| Dataset             | Model              | ERDE5 ↓ | ERDE50 ↓ | F1 ↑   |
|---------------------|--------------------|---------|----------|--------|
| **Depression 2017**  | LabChain TWEC_SVM  | **12.44** | **7.66**  | **0.50** |
|                     | original TWEC_SVM  | 12.56   | 10.21    | 0.25   |
| **Depression 2018**  | LabChain TWEC_SVM  | 9.36    | **5.56** | **0.46** |
|                     | original TWEC_SVM  | **9.09**| 7.84     | 0.21   |
| **Depression 2022**  | LabChain TWEC_SVM  | **6.57** | **3.81** | **0.46** |
|                     | original TWEC_SVM  | 6.69    | 5.82     | 0.17   |
| **Anorexia 2018**    | LabChain TWEC_SVM  | **18.35**| **12.52**| **0.26** |
|                     | original TWEC_SVM  | 21.40   | 15.23    | 0.22   |
| **Anorexia 2019**    | LabChain TWEC_SVM  | **7.95** | **2.60** | **0.63** |
|                     | original TWEC_SVM  | 8.17    | 6.30     | 0.27   |
| **Self-harm 2020**   | LabChain TWEC_SVM  | **19.96**| 9.17     | **0.62** |
|                     | original TWEC_SVM  | 18.79   | **16.37**| 0.41   |
| **Self-harm 2021**   | LabChain TWEC_SVM  | 9.80    | **5.42** | **0.33** |
|                     | original TWEC_SVM  | **9.02**| 7.21     | 0.26   |
| **Gambling 2022**    | LabChain TWEC_SVM  | **3.35** | **0.88** | **0.38** |
|                     | original TWEC_SVM  | 3.76    | 3.31     | 0.13   |
| **Gambling 2023**    | LabChain TWEC_SVM  | 2.96    | **0.41** | **0.88** |
|                     | original TWEC_SVM  | **2.92**| 1.10     | 0.74   |

*Note:* Lower ERDE scores indicate better early detection performance, while higher F1 scores indicate better classification accuracy.

---

## 📁 Project Structure

```
📁 Temporal-Word-Embeddings
├── data
├── notebooks
└── src
    ├── datasets
    ├── filters
    ├── metrics
    └── models
```

---

## 📄 Citation

If you use our work, please cite it as follows:

```bibtex
@article{couto2025temporal,
  title={Temporal Word Embeddings for Early Detection of Psychological Disorders on Social Media},
  author={Couto, Manuel and Perez, Anxo and Parapar, Javier and Losada, David E},
  journal={Journal of Healthcare Informatics Research},
  pages={1--30},
  year={2025},
  publisher={Springer}
}
```

---

## 📜 References

> Manuel Couto, Anxo Perez, Javier Parapar & David E. Losada  (2025). *Temporal Word Embeddings for Early Detection of Psychological Disorders on Social Media*. Journal of Healthcare Informatics Research. [https://doi.org/10.1007/s41666-025-00186-9](https://link.springer.com/article/10.1007/s41666-025-00186-9)

---

## 📫 Contact

For questions or collaboration, contact:
✉️ [manuel.couto.pintos@usc.es](mailto:manuel.couto.pintos@usc.es)
🔗 [LinkedIn](https://www.linkedin.com/in/manuelcoutopintos/)

