# Qualite des donnees

## Source
- Fournisseur: Copernicus Climate Data Store (CDS)
- Jeu de donnees: reanalysis-era5-single-levels-monthly-means
- Produit: monthly_averaged_reanalysis
- Periode: 2000-01 a 2020-12
- Zone: CORDEX-Africa [43, -25, -45, 60]

## Variables telechargees
- 2m_temperature
- total_precipitation
- 10m_u_component_of_wind
- 10m_v_component_of_wind
- surface_solar_radiation_downwards
- mean_sea_level_pressure

## Resolution
- Temporelle: mensuelle (1 pas par mois)
- Spatiale: grille ERA5 native, puis regrillee a 0.5 deg (lat/lon) avec xesmf (bilinear)

## Controle qualite effectue
- Verification doublons temporels
- Tri chronologique de l'axe time
- Remplacement des valeurs sentinelles (seuil absolu > 1e15)
- Interpolation lineaire des petits trous temporels (max 3 pas)
- Standardisation des coordonnees vers lat/lon

## Taux de valeurs manquantes
- Methode: scan echantillonne sur t=0 par fichier et par variable
- Resultat: voir le tableau `mv_df` du notebook et les exports dans outputs/
- Alerte: variables avec taux > 5% signalees dans le notebook

## Traçabilite
- Parametres centralises dans le bloc CONFIG en tete du notebook
- Rapport de reproductibilite exporte: outputs/reproducibility_report.json
