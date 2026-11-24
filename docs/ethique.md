# Réflexion éthique

## 1. Détails inventés

Un modèle de super-résolution ne retrouve pas l'information perdue : il la reconstruit à partir de ce qu'il a appris. Les modèles génératifs (GAN, diffusion) produisent des images plus nettes, mais les détails ajoutés peuvent être faux : motifs, textures, caractères ou traits du visage plausibles mais inexistants dans la scène d'origine. Un PSNR plus bas accompagné d'un meilleur LPIPS peut ainsi traduire une image plus agréable mais moins fidèle.

Mesure prévue : un catalogue des cas où le modèle invente des structures, avec comparaison à l'image de référence (`results/`).

## 2. Risques d'usage

| Risque | Exemple | Mesure |
|---|---|---|
| Fausse preuve | Image de surveillance agrandie présentée comme fidèle | Mention explicite : l'image agrandie est une estimation |
| Erreur d'interprétation | Imagerie médicale ou documents agrandis | Usage déconseillé dans la fiche du modèle |
| Désinformation | Image améliorée diffusée sans mention | Marquer les sorties de la démonstration comme générées |

## 3. Données

- Les images proviennent de jeux de données publics à usage de recherche (voir `docs/licences.md`).
- Aucune donnée personnelle n'est collectée par le projet.
- Le jeu d'entraînement est petit et limité à des images naturelles : les performances ne sont pas garanties sur d'autres types d'images ni sur des populations ou contextes différents.

## 4. Coût environnemental

L'entraînement est volontairement léger : réseau de petite taille, quelques dizaines de minutes de GPU T4. Les expériences supplémentaires augmentent la consommation d'énergie : les essais sont limités au nécessaire.

## 5. Transparence

- Résultats, limites et conditions d'évaluation documentés dans le README et la fiche du modèle.
- Dépôt public : code, configurations et mesures reproductibles.