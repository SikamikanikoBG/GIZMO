WIP :)

For building sphinx look in ``./conda_envs``

For building sphinx doc look in ``./docs/auto_make_html.sh``

### Multiclass predictions with Gizmo

Now it is possible to build a multiclass models with Scoring Gizmo.
A multiclass model is one that targets more than two classes in its criterion (label / target) variable.

There's nothing special / different in using multiclass modeling: you specify the criterion variable in the project's params file, and the Scoring Gizmo package takes care of processing the data.
In the evaluation session it will create ROC graphs that capture all classes' performance.


There are three main differences between the binary classification case and the multiclass case:
- Only XGBoost is used right now. It is possible to use other algorithms, by means of creating one-vs-rest datasets, but that would require significant re-engineering effort for the whole package.
- Feature selection: In the binary classification case, feature selection is mainly done by calculating correlations between the predictors and the target. In the case of multiclass criterion variable we use **predictive power score**  to do feature selection. This methodology results in **far less features selected**, but it is possible to capture non-linear and/or asymetrical relationships between the predictor variables. In the case of correlation, it will always capture linear and symetrical relationships only.
- Probabilities and bands. As we have multiple classes, defining cut-offs for those classes is hard, and it is non-trivial to implement in a generic way.