# Surreal DB Import

You can install [SurrealDB](https://surrealdb.com) in diverse ways and the [Surrealist](https://surrealdb.com/surrealist) is a handy way to manage Surreal data.
Via Surrealist you can spin up a playground instance to import a Knwler extraction. If you prefer a more permananent database you can also use Docker:

```bash
docker run -d --name surreal --pull always \
  -p 8800:8800 \
  -v ~/surreal:/data \
  surrealdb/surrealdb:latest \
  start --log debug --user root --pass root \
  --bind 0.0.0.0:8800 rocksdb:/data/trigger.db
```

If you installed it via `brew` you can start an instance like so:

```bash
surreal start --user root --pass root rocksdb:~/surreal/
```
Change the data volume or database directory as needed. Surreal has an usual [namespace/database/user](https://surrealdb.com/docs/surrealdb/introduction/concepts#namespaces-and-databases) system, make sure you log in with the correct channel.

From here on we'll assume that Surreal runs on port 8000 and the login is `root/root`. If you want to completely wipe the content of the database you can simply delete the data dir and restart the server since Surreal does not have a global 'clear'.





