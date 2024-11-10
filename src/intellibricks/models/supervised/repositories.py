# cortex/core/models/repositories/impl.py

import json
import os
from typing import Any, Optional

from weavearc.data import (
    AsyncRepository,
    CreateResult,
    ReadAllResult,
    ReadResult,
    UpdateResult,
    DeleteResult,
)

from ...data.entities import ForgedModel


class LocalSupervisedModelRepository(AsyncRepository[ForgedModel]):
    """
    Local repository implementation for storing and retrieving forged models.

    Models are stored in the local filesystem under 'cortex/core/models/store/{uid}/model_info.json'.
    """

    async def create(
        self, entity: ForgedModel, *, filters: Optional[dict[str, Any]] = None
    ) -> CreateResult:
        model_dir = os.path.join(
            "cortex", "core", "models", "supervised", "store", entity.uid
        )
        os.makedirs(model_dir, exist_ok=True)
        model_info_path = os.path.join(model_dir, "model_info.json")
        with open(model_info_path, "w") as f:
            json.dump(entity.as_dict(), f, indent=4)
        return CreateResult(uid=entity.uid)

    async def read(
        self, q: str, *, filters: Optional[dict[str, Any]] = None
    ) -> ReadResult[ForgedModel]:
        model_dir = os.path.join("cortex", "core", "models", "supervised", "store", q)
        model_info_path = os.path.join(model_dir, "model_info.json")
        if not os.path.exists(model_info_path):
            raise ModuleNotFoundError

        with open(model_info_path, "r") as f:
            model_info = json.load(f)
        model = ForgedModel.from_dict(model_info)
        return ReadResult(entity=model)

    async def read_all(
        self, *, filters: Optional[dict[str, Any]] = None
    ) -> ReadAllResult[ForgedModel]:
        models = []
        models_dir = os.path.join("cortex", "core", "models", "supervised", "store")
        for model_dir in os.listdir(models_dir):
            model_info_path = os.path.join(models_dir, model_dir, "model_info.json")
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
            model = ForgedModel.from_dict(model_info)
            models.append(model)

        return ReadAllResult(entities=models)

    async def update(
        self, q: str, entity: ForgedModel, *, filters: Optional[dict[str, Any]] = None
    ) -> UpdateResult:
        raise NotImplementedError

    async def delete(
        self, q: str, *, filters: Optional[dict[str, Any]] = None
    ) -> DeleteResult:
        raise NotImplementedError
