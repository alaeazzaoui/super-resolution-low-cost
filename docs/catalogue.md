# Catalogue des détails inventés ou perdus

## 1. Méthode

- Images : Set5 et Set14, facteur ×4, version basse résolution créée par réduction bicubique.
- Pour chaque image, la zone zoomée est choisie automatiquement : fenêtre de 64 × 64 pixels où l'erreur absolue du dernier modèle est maximale. Les zooms présentent donc le pire cas de chaque image, pas le comportement moyen.
- Les figures complètes sont dans `results/catalogue/`.

## 2. Types de défauts observés

| Type | Description |
|---|---|
| Texture lissée | La structure générale est retrouvée, mais le détail fin reste flou |
| Détail perdu | Un élément détruit par la réduction n'est pas restitué |
| Structure inexacte | Le modèle produit des formes nettes mais différentes de la référence |

## 3. Cas observés (EDSR léger)

| Image | PSNR bicubique | PSNR EDSR | Zone | Type | Observation |
|---|---|---|---|---|---|
| `Set5_baby` | 30,64 | 31,58 | Tricot du bonnet | Texture lissée | Mailles verticales retrouvées mais moins contrastées que la référence ; fibres fines non restituées |
| `Set5_bird` | 28,33 | 29,37 | Œil du toucan | Détail perdu | Contours plus nets, mais le reflet blanc de la pupille est absent dans le bicubique comme dans EDSR |
| `Set5_butterfly` | 21,10 | 22,43 | Points de la base de l'aile | Structure inexacte | Taches irrégulières remplacées par des formes plus rondes en anneaux ; erreur concentrée autour de chaque tache |

## 4. Cas du GAN

À compléter après l'entraînement du GAN : comparer, sur les mêmes images, la netteté gagnée et les structures ajoutées par rapport à la référence.

## 5. Conclusion provisoire

Avec une perte L1, EDSR léger corrige le flou sans ajouter de détails visibles : les défauts sont des textures lissées, des détails perdus et quelques structures inexactes sur les motifs répétitifs fins. Le risque de détails inventés devrait être plus élevé avec le GAN.