from typing import List

from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.storage import LocalStorage, Storage


class ArtifactRegistry:
    def __init__(self, database: Database, storage: Storage):
        """Registry that can store Artifacts.

        Args:
            database (Database): Database to store the Artifacts in RAM
            storage (Storage): Storage to store the Artifacts on the disk
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
        """Register an artifact into the registry.

        Args:
            artifact (Artifact): Artifact to register. It's asset path should
            be collection/name_of_artifact. (TODO Confirm this!)
        """
        # GW: NOTE I don't like this implementation. Database.set should be
        # responsible for saving the data into the storage.
        # Save the artifact in the storage.
        self._storage.save(artifact.data, artifact.asset_path)
        # Save the metadata in the database.
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set(f"artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """Get a list of all stored Artifacts. Optinally get a list of the
        artifacts of a specified type.

        Args:
            type (str, optional): Type of Arfifacts to list. If none print all
            Artifacts.

        Returns:
            List[Artifact]: List of stored artifacts (with type `type' if
            specified.)
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str):
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo")),
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self):
        return self._registry
