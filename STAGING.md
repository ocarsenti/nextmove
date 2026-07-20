# Environnement de staging — NextMove

Ce document décrit l'environnement de staging, distinct de la production, sur le VPS.

## Principe

- **Prod** : dossier/service/port existants, non touchés par ce qui suit.
- **Staging** : clone séparé du repo, sur sa propre branche de travail, son propre port,
  son propre service systemd et son propre `.env`.

## Mise en place initiale (à faire une fois, sur le VPS)

```bash
sudo mkdir -p /opt/nextmove-staging
sudo chown $USER:$USER /opt/nextmove-staging
git clone -b v5-matching https://github.com/ocarsenti/nextmove.git /opt/nextmove-staging
cd /opt/nextmove-staging
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Créer `/opt/nextmove-staging/.env` (séparé du `.env` de prod — ne jamais partager les clés) :

```
ANTHROPIC_API_KEY=...   # peut être une clé distincte de la prod si tu veux isoler la conso
NEXTMOVE_PORT=8001      # prod reste sur son port habituel (ex. 8000)
```

Créer le service systemd `/etc/systemd/system/nextmove-staging.service` :

```ini
[Unit]
Description=NextMove Staging
After=network.target

[Service]
WorkingDirectory=/opt/nextmove-staging
EnvironmentFile=/opt/nextmove-staging/.env
ExecStart=/opt/nextmove-staging/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8001
Restart=on-failure
User=%i

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nextmove-staging
```

**Frontend** : vérifie que `frontend/index_v5.html` (version staging) pointe vers
`http://<vps>:8001` et non vers le port de prod — un mélange des deux s'est déjà produit
sur ce projet par le passé.

## Déployer une nouvelle version sur staging

Depuis le VPS :

```bash
cd /opt/nextmove-staging
./deploy_staging.sh
```

Le script :
1. récupère la dernière version de la branche (`v5-matching` par défaut),
2. installe les dépendances,
3. **lance la suite pytest complète** (`make test` / `pytest.ini`, seuil de couverture 75%),
4. ne redémarre le service que si les tests passent,
5. vérifie que le service est bien actif après redémarrage.

Variables d'environnement optionnelles pour surcharger les valeurs par défaut :
`NEXTMOVE_STAGING_DIR`, `NEXTMOVE_STAGING_BRANCH`, `NEXTMOVE_STAGING_SERVICE`.

## Logs

```bash
journalctl -u nextmove-staging -f
```

## Revenir en arrière

```bash
cd /opt/nextmove-staging
git log --oneline -10        # repérer le commit précédent
git reset --hard <commit>
sudo systemctl restart nextmove-staging
```

## Ce qui manque encore

- Pas de persistance des jobs (`_job_store` est en mémoire, comme en prod) : un restart
  de staging vide les job cards créées en test. Acceptable pour du staging, à surveiller
  si ça devient gênant.
- Le script suppose un utilisateur avec droits `sudo systemctl` sans mot de passe pour
  `nextmove-staging` spécifiquement — à configurer via `visudo` si ce n'est pas déjà le cas.

## Production — http://54.38.26.33/nextmove-v5/

Contrairement au staging ci-dessus (`/opt/nextmove-staging`, port 8001), la prod tourne
directement dans le clone historique du repo sur le VPS :

- Répertoire : `/home/olive/NEXTMOVE-V5` (branche `v5-matching`)
- Service systemd : `nextmove-v5-matching-api`
  (`ExecStart=/home/olive/NEXTMOVE-V5/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8096`)
- Nginx : bloc `server_name 54.38.26.33` dans `/etc/nginx/sites-available/docalib_demo`
  - `/nextmove-v5/` → alias statique sur `frontend/index_v5.html`
  - `/nextmove-v5-api/` → proxy vers `127.0.0.1:8096`
  - `/nextmove-v5-api/study/{retest,quality}/report` et `/nextmove-v5/admin_retest.html`
    sont protégés par Basic Auth (`.htpasswd_nextmove_admin`)
- Logs : `/home/olive/NEXTMOVE-V5/api_server.log` (le service redirige stdout/stderr dedans,
  pas de `journalctl` applicatif — `journalctl -u nextmove-v5-matching-api` ne donne que le
  cycle de vie du process)

### Déployer une nouvelle version en production

Depuis le VPS :

```bash
cd /home/olive/NEXTMOVE-V5
./deploy_prod.sh
```

Le script (même logique que `deploy_staging.sh`, adapté aux chemins prod — `venv/` et non
`.venv/`, pas de `/opt`) :
1. récupère la dernière version de la branche (`v5-matching` par défaut),
2. installe les dépendances,
3. lance la suite pytest complète (seuil de couverture 75%),
4. ne redémarre `nextmove-v5-matching-api` que si les tests passent,
5. vérifie que le service est bien actif après redémarrage.

Variables d'environnement optionnelles : `NEXTMOVE_PROD_DIR`, `NEXTMOVE_PROD_BRANCH`,
`NEXTMOVE_PROD_SERVICE`.

Le redémarrage utilise `sudo systemctl restart nextmove-v5-matching-api`, autorisé sans
mot de passe via `/etc/sudoers.d/nextmove-deploy` (règle scoping restreinte à
`systemctl restart|status nextmove-v5-matching-api`, configurée le 2026-07-20).

### Revenir en arrière (prod)

```bash
cd /home/olive/NEXTMOVE-V5
git log --oneline -10
git reset --hard <commit>
sudo systemctl restart nextmove-v5-matching-api
```
