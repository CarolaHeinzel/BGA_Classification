Vorgehen:

1. Verstehe die Anwendung von TabPFN auf die Daten in filtered_population_eur.xlsx
2. Wandle die Daten in den .vcf.recode Dateien um, sodass wir TabPFN darauf anwenden können
3. Wähle aus den 22 .vcf.recode dateien insgesamt zufällig 500 und 104 features (marker) aus + vergleiche die Ergebnisse mit denen aus filtered_population_eur.xlsx 
4. Idealerweise: wähle die Marker nicht mehr zufällig aus sonder nach den Differenzen zwischen den Allelfrequenzen (darüber können wir auch noch mal reden) <br>
5. Verstehe den Code zur Feature Selection und wende ihn an
6. Literaturübersicht: Feature selection in der Forensik (damit habe ich schon angefangen) und im Allgemeinen.

Daten: 
1) filtered_population_eur.xlsx (British in England and Scotland, Spain, Toscani in Italy, Finland)
2) .vcf.recode (alle Individuen +  alle Marker)

Vergleiche 1) und 2) mit Crossvalidation und TabPFN, sklearn Naive Bayes mit Log Loss, Roc Auc, Accuracy and Confsion Matrix. wichtig, dass immer die gleichen Individuen pro CV verwendet werden!

Diese beiden links werden dann später relevant sein:
LEI (Könnte relevant sein):
https://www.nature.com/articles/s41598-019-47012-y

Link zu SHAP für TabPFN:

https://www.linkedin.com/posts/maximilian-muschalik_native-tabpfn-support-in-%F0%9D%98%80%F0%9D%97%B5%F0%9D%97%AE%F0%9D%97%BD%F0%9D%97%B6%F0%9D%97%BE-via-activity-7285223644736188416-FxRw?utm_source=share&utm_medium=member_desktop

