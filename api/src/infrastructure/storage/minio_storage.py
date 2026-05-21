from datetime import timedelta
from io import BytesIO

from minio import Minio

from src.core.settings import settings


class MinIOStorage:
    __instance = None
    _client: Minio = None
    _public_client: Minio = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls._client = Minio(
                endpoint=settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            # region pre-seeds the SDK so it skips the GetBucketLocation HTTP
            # call — without it the SDK would try to connect to the public
            # endpoint from inside the container and fail.
            cls._public_client = Minio(
                endpoint=settings.MINIO_PUBLIC_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
                region=settings.MINIO_REGION,
            )
        return cls.__instance

    def _ensure_bucket(self):
        if not self._client.bucket_exists(settings.MINIO_BUCKET):
            self._client.make_bucket(settings.MINIO_BUCKET)

    def upload(
        self,
        object_name: str,
        data: bytes,
        content_type: str = 'application/pdf',
    ) -> str:
        """Upload bytes to MinIO and return the object name.

        Args:
            object_name (str): Name of the file in the bucker.
            data (bytes): The file bytes.
            content_type (str): The file content type.

        Returns:
            str: Uploaded object name
        """
        self._ensure_bucket()
        self._client.put_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return object_name

    def presigned_url(self, bucket: str, object_name: str) -> str:
        """Generate a 5-minutes presigned URL via the public endpoint.

        Args:
            bucket (str): Name of the bucket.
            object_name (str): Name of the file in the bucker.

        Returns:
            str: Presigned URL
        """
        return self._public_client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_name,
            expires=timedelta(minutes=5),
        )
