from __future__ import annotations

import base64
import hashlib
import importlib
import mimetypes
import sys
import zipfile
from enum import Enum
from http.cookiejar import CookieJar
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    TypeAlias,
)

import msgspec
import requests
import validators
from _typeshed import SupportsItems, SupportsRead
from requests import Response
from requests.auth import AuthBase
from requests.models import PreparedRequest
from typing_extensions import Self

_TextMapping: TypeAlias = MutableMapping[str, str]
_HeadersMapping: TypeAlias = Mapping[str, str | bytes | None]

_Data: TypeAlias = (
    # used in requests.models.PreparedRequest.prepare_body
    #
    # case: is_stream
    # see requests.adapters.HTTPAdapter.send
    # will be sent directly to http.HTTPConnection.send(...) (through urllib3)
    Iterable[bytes]
    # case: not is_stream
    # will be modified before being sent to urllib3.HTTPConnectionPool.urlopen(body=...)
    # see requests.models.RequestEncodingMixin._encode_params
    # see requests.models.RequestEncodingMixin._encode_files
    # note that keys&values are converted from Any to str by urllib.parse.urlencode
    | str
    | bytes
    | SupportsRead[str | bytes]
    | list[tuple[Any, Any]]
    | tuple[tuple[Any, Any], ...]
    | Mapping[Any, Any]
)

_ParamsMappingKeyType: TypeAlias = str | bytes | int | float
_ParamsMappingValueType: TypeAlias = (
    str | bytes | int | float | Iterable[str | bytes | int | float] | None
)
_Params: TypeAlias = (
    SupportsItems[_ParamsMappingKeyType, _ParamsMappingValueType]
    | tuple[_ParamsMappingKeyType, _ParamsMappingValueType]
    | Iterable[tuple[_ParamsMappingKeyType, _ParamsMappingValueType]]
    | str
    | bytes
)
_Verify: TypeAlias = bool | str
_Timeout: TypeAlias = float | tuple[float, float] | tuple[float, None]
_Cert: TypeAlias = str | tuple[str, str]
Incomplete: TypeAlias = Any
_Hook: TypeAlias = Callable[[Response], Any]
_HooksInput: TypeAlias = Mapping[str, Iterable[_Hook] | _Hook]
_FileContent: TypeAlias = SupportsRead[str | bytes] | str | bytes
_FileName: TypeAlias = str | None
_FileContentType: TypeAlias = str
_FileSpecTuple2: TypeAlias = tuple[_FileName, _FileContent]
_FileSpecTuple3: TypeAlias = tuple[_FileName, _FileContent, _FileContentType]
_FileCustomHeaders: TypeAlias = Mapping[str, str]
_FileSpecTuple4: TypeAlias = tuple[
    _FileName, _FileContent, _FileContentType, _FileCustomHeaders
]
_FileSpec: TypeAlias = (
    _FileContent | _FileSpecTuple2 | _FileSpecTuple3 | _FileSpecTuple4
)
_Files: TypeAlias = Mapping[str, _FileSpec] | Iterable[tuple[str, _FileSpec]]
_Auth: TypeAlias = (
    tuple[str, str] | AuthBase | Callable[[PreparedRequest], PreparedRequest]
)


if TYPE_CHECKING:
    from fastapi import UploadFile as FastAPIUploadFile
    from litestar.datastructures import UploadFile as LitestarUploadFile


class FileExtension(str, Enum):
    PDF = "pdf"
    JSON = "json"
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    HTML = "html"
    TXT = "txt"
    # Add more extensions as needed


class RawFile(msgspec.Struct, frozen=True, gc=False):
    """
    Represents an immutable raw file with its content and extension.

    The `RawFile` class is designed for efficient and immutable handling of raw file data.
    It stores file contents as immutable bytes and provides utility methods for reading,
    writing, and manipulating the file content without mutating the original data.

    **Key Features:**

    - **Immutability**: Instances of `RawFile` are immutable. Once created, their contents cannot be modified.
      This is enforced by using `msgspec.Struct` with `frozen=True`, ensuring thread-safety and predictability.
    - **Performance**: Optimized for speed and memory efficiency by disabling garbage collection (`gc=False`)
      and using immutable data structures. This reduces overhead and can significantly boost performance,
      especially when handling many instances.
    - **Compactness**: Stores file content in memory as bytes, leading to fast access and manipulation.
      The absence of mutable state allows for leaner objects.
    - **Garbage Collection**: By setting `gc=False`, the class instances are excluded from garbage collection tracking.
      This improves performance when creating many small objects but requires careful management of resources.
    - **Compression Support**: Provides methods for compressing and decompressing file contents using gzip,
      returning new `RawFile` instances without altering the original data.
    - **Versatile Creation Methods**: Offers multiple class methods to create `RawFile` instances from various sources,
      such as file paths, bytes, base64 strings, strings, streams, URLs, and cloud storage services.

    **Important Notes:**

    - **Memory Usage**: Since the entire file content is stored in memory, handling very large files may lead
      to high memory consumption. Ensure that file sizes are manageable within the available system memory.
    - **Resource Management**: As garbage collection is disabled, it's crucial to manage resources appropriately.
      While the class is designed to be immutable and not require cleanup, be cautious when handling external resources.
    - **Thread-Safety**: Immutability ensures that instances of `RawFile` are inherently thread-safe.

    **Example Usage:**

    ```python
    # Create a RawFile instance from a file path
    raw_file = RawFile.from_file_path('example.pdf')

    # Access the file extension
    print(raw_file.extension)  # Output: FileExtension.PDF

    # Get the size of the file content
    print(raw_file.get_size())  # Output: Size of the file in bytes

    # Compute checksums
    md5_checksum = raw_file.compute_md5()
    sha256_checksum = raw_file.compute_sha256()

    # Save the content to a new file
    raw_file.save_to_file('copy_of_example.pdf')

    # Compress the file content
    compressed_file = raw_file.compress()

    # Decompress the file content
    decompressed_file = compressed_file.decompress()
    ```

    **Methods Overview:**

    - Creation:
      - `from_file_path(cls, file_path: str)`: Create from a file path.
      - `from_bytes(cls, data: bytes, extension: FileExtension)`: Create from bytes.
      - `from_base64(cls, b64_string: str, extension: FileExtension)`: Create from a base64 string.
      - `from_string(cls, content: str, extension: FileExtension, encoding: str = "utf-8")`: Create from a string.
      - `from_stream(cls, stream: BinaryIO, extension: FileExtension)`: Create from a binary stream.
      - `from_url(cls, url: str, ...)`: Create from a URL.
      - `from_s3(cls, bucket_name: str, object_key: str, extension: Optional[FileExtension] = None)`: Create from Amazon S3.
      - `from_azure_blob(cls, connection_string: str, container_name: str, blob_name: str, extension: Optional[FileExtension] = None)`: Create from Azure Blob Storage.
      - `from_gcs(cls, bucket_name: str, blob_name: str, extension: Optional[FileExtension] = None)`: Create from Google Cloud Storage.
      - `from_zip(cls, zip_file_path: str, inner_file_path: str, extension: Optional[FileExtension] = None)`: Create from a file within a ZIP archive.
      - `from_stdin(cls, extension: FileExtension)`: Create from standard input.

    - Utilities:
      - `save_to_file(self, file_path: str)`: Save content to a file.
      - `get_size(self) -> int`: Get the size of the content in bytes.
      - `compute_md5(self) -> str`: Compute MD5 checksum.
      - `compute_sha256(self) -> str`: Compute SHA256 checksum.
      - `get_mime_type(self) -> str`: Get MIME type based on the file extension.
      - `compress(self) -> RawFile`: Compress content using gzip.
      - `decompress(self) -> RawFile`: Decompress gzip-compressed content.
      - `read_async(self) -> bytes`: Asynchronously read the content.

    **Immutability Enforcement:**

    - The class is decorated with `msgspec.Struct` and `frozen=True`, which makes all instances immutable.
    - Any method that would traditionally modify the instance returns a new `RawFile` instance instead.
    - This design ensures that the original data remains unchanged, promoting safer and more predictable code.

    **Performance Considerations:**

    - **No Garbage Collection Overhead**: By setting `gc=False`, instances are not tracked by the garbage collector, reducing overhead.
      This is suitable when instances do not contain cyclic references.
    - **Optimized Data Structures**: Using immutable bytes and avoiding mutable state enhances performance and reduces memory footprint.
    - **Fast Access**: In-memory storage allows for rapid access and manipulation of file content.

    **Garbage Collection and Resource Management:**

    - While garbage collection is disabled for instances, Python's reference counting will still deallocate objects when they are no longer in use.
    - Be mindful when working with external resources (e.g., open files or network connections) to ensure they are properly closed.
    - Since `RawFile` instances hold data in memory, they are automatically cleaned up when references are removed.

    **Thread-Safety:**

    - Immutable objects are inherently thread-safe because their state cannot change after creation.
    - `RawFile` instances can be shared across threads without the need for synchronization mechanisms.

    **Compression Level:**

    - The `compress` and `decompress` methods use gzip with default compression levels.
    - If you need to specify a compression level, you can modify the methods to accept a parameter for the compression level.

    **Extensibility:**

    - The `FileExtension` enum and content type mappings can be extended to support additional file types as needed.
    - Custom methods can be added to handle specific use cases or integrations with other services.

    **Examples of Creating `RawFile` Instances from Different Sources:**

    ```python
    # From bytes
    raw_file = RawFile.from_bytes(b"Hello, World!", FileExtension.TXT)

    # From a base64 string
    raw_file = RawFile.from_base64("SGVsbG8sIFdvcmxkIQ==", FileExtension.TXT)

    # From a URL
    raw_file = RawFile.from_url("https://example.com/data.json")

    # From Amazon S3
    raw_file = RawFile.from_s3("my-bucket", "path/to/object.json")

    # From Azure Blob Storage
    raw_file = RawFile.from_azure_blob("connection_string", "container", "blob.json")

    # From Google Cloud Storage
    raw_file = RawFile.from_gcs("my-bucket", "path/to/blob.json")

    # From standard input
    raw_file = RawFile.from_stdin(FileExtension.TXT)
    ```

    **Disclaimer:**

    - Ensure that all necessary dependencies are installed for methods that interface with external services.
    - Handle exceptions appropriately in production code, especially when dealing with I/O operations and network requests.
    """

    contents: bytes
    extension: FileExtension

    @classmethod
    def from_file_path(cls: type[RawFile], file_path: str) -> RawFile:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        if not path.is_file():
            raise ValueError(f"{file_path} is not a file")

        with open(file_path, "rb") as f:
            data = f.read()

        return cls(
            contents=data,
            extension=FileExtension(path.suffix.lstrip(".")),
        )

    @classmethod
    def from_bytes(cls, data: bytes, extension: FileExtension) -> RawFile:
        return cls(contents=data, extension=extension)

    @classmethod
    def from_base64(cls, b64_string: str, extension: FileExtension) -> RawFile:
        data = base64.b64decode(b64_string)
        return cls.from_bytes(data, extension)

    @classmethod
    def from_string(
        cls, content: str, extension: FileExtension, encoding: str = "utf-8"
    ) -> RawFile:
        data = content.encode(encoding)
        return cls.from_bytes(data, extension)

    @classmethod
    def from_stream(cls, stream: BinaryIO, extension: FileExtension) -> RawFile:
        data = stream.read()
        return cls(contents=data, extension=extension)

    @classmethod
    def from_litestar_upload_file(
        cls: type[RawFile], file: LitestarUploadFile
    ) -> RawFile:
        loader = importlib.find_loader("litestar")
        if loader is None:
            raise ImportError(
                """
                Litestar is required to use this method. Please install it with:
                >>> pip install litestar
                """
            )

        extension: Optional[FileExtension] = cls._get_extension_from_content_type(
            file.content_type
        )

        if extension is None:
            raise ValueError(f"{file.content_type} is not a supported file type yet.")

        data = file.file.read()
        return cls(contents=data, extension=extension)

    @classmethod
    def from_fastapi_upload_file(
        cls: type[RawFile], file: FastAPIUploadFile
    ) -> RawFile:
        loader = importlib.find_loader("fastapi")
        if loader is None:
            raise ImportError(
                """
                FastAPI is required to use this method. Please install it with:
                >>> pip install fastapi
                """
            )

        if file.content_type is None:
            raise ValueError("The content type of the file is missing.")

        extension: Optional[FileExtension] = cls._get_extension_from_content_type(
            file.content_type
        )

        if extension is None:
            raise ValueError(f"{file.content_type} is not a supported file type yet.")

        data = file.file.read()
        return cls(contents=data, extension=extension)

    @classmethod
    def from_url(
        cls: type[Self],
        url: str | bytes,
        *,
        params: Optional[_Params] = None,
        data: Optional[_Data] = None,
        headers: Optional[_HeadersMapping] = None,
        cookies: Optional[CookieJar | _TextMapping] = None,
        files: Optional[_Files] = None,
        auth: Optional[_Auth] = None,
        timeout: Optional[_Timeout] = None,
        allow_redirects: bool = False,
        proxies: Optional[_TextMapping] = None,
        hooks: Optional[_HooksInput] = None,
        stream: Optional[bool] = None,
        verify: Optional[_Verify] = None,
        cert: Optional[_Cert] = None,
        json: Optional[Incomplete] = None,
        extension: Optional[FileExtension] = None,
    ) -> RawFile:
        validators.url(url)

        response: requests.Response = requests.get(
            url,
            params=params,
            data=data,
            headers=headers,
            cookies=cookies,
            files=files,
            auth=auth,
            timeout=timeout,
            allow_redirects=allow_redirects,
            proxies=proxies,
            hooks=hooks,
            stream=stream,
            verify=verify,
            cert=cert,
            json=json,
        )
        response.encoding = "utf-8"

        response_content: bytes | Any = response.content

        if not isinstance(response_content, bytes):
            data = str(data).encode("utf-8")

        file_extension = extension or (
            cls._get_extension_from_content_type(
                response.headers.get("Content-Type", "").split(";")[0]
            )
            or FileExtension.HTML
        )

        return cls(contents=response_content, extension=file_extension)

    @classmethod
    def from_s3(
        cls,
        bucket_name: str,
        object_key: str,
        extension: Optional[FileExtension] = None,
    ) -> RawFile:
        loader = importlib.find_loader("boto3")
        if loader is None:
            raise ImportError(
                """
                Boto3 is required to use this method. Please install it with:
                >>> pip install boto3
                """
            )

        import boto3

        s3 = boto3.client("s3")

        if not extension:
            ext = Path(object_key).suffix.lstrip(".")
            if ext.upper() in FileExtension.__members__:
                extension = FileExtension[ext.upper()]
            else:
                head_object = s3.head_object(Bucket=bucket_name, Key=object_key)
                content_type = head_object.get("ContentType", "")
                extension = cls._get_extension_from_content_type(content_type)

        if not extension:
            raise ValueError(
                "Unable to determine the file extension. Please specify it explicitly."
            )

        obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        data = obj["Body"].read()

        return cls(contents=data, extension=extension)

    @classmethod
    def from_azure_blob(
        cls,
        connection_string: str,
        container_name: str,
        blob_name: str,
        extension: Optional[FileExtension] = None,
    ) -> RawFile:
        loader = importlib.find_loader("azure.storage.blob")
        if loader is None:
            raise ImportError(
                """
                Azure SDK is required to use this method. Please install it with:
                >>> pip install azure-storage-blob
                """
            )

        from azure.storage.blob import BlobServiceClient

        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        if not extension:
            ext = Path(blob_name).suffix.lstrip(".")
            if ext.upper() in FileExtension.__members__:
                extension = FileExtension[ext.upper()]
            else:
                properties = blob_client.get_blob_properties()
                content_type = properties.content_settings.content_type
                if content_type is None:
                    raise ValueError(
                        "Unable to determine the file extension. Please specify it explicitly."
                    )

                extension = cls._get_extension_from_content_type(content_type)

        if not extension:
            raise ValueError(
                "Unable to determine the file extension. Please specify it explicitly."
            )

        stream = blob_client.download_blob()
        data = stream.readall()

        return cls(contents=data, extension=extension)

    @classmethod
    def from_gcs(
        cls, bucket_name: str, blob_name: str, extension: Optional[FileExtension] = None
    ) -> RawFile:
        loader = importlib.find_loader("google.cloud.storage")
        if loader is None:
            raise ImportError(
                """
                Google Cloud Storage is required to use this method. Please install it with:
                >>> pip install google-cloud-storage
                """
            )

        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not extension:
            ext = Path(blob_name).suffix.lstrip(".")
            if ext.upper() in FileExtension.__members__:
                extension = FileExtension[ext.upper()]
            else:
                blob.reload()
                content_type = blob.content_type
                extension = cls._get_extension_from_content_type(content_type)

        if not extension:
            raise ValueError(
                "Unable to determine the file extension. Please specify it explicitly."
            )

        data = blob.download_as_bytes()

        return cls(contents=data, extension=extension)

    @classmethod
    def from_zip(
        cls,
        zip_file_path: str,
        inner_file_path: str,
        extension: Optional[FileExtension] = None,
    ) -> RawFile:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            with zip_ref.open(inner_file_path) as file:
                data = file.read()
                if not extension:
                    extension = FileExtension(Path(inner_file_path).suffix.lstrip("."))
                return cls(contents=data, extension=extension)

    @classmethod
    def from_database_blob(cls, blob_data: bytes, extension: FileExtension) -> RawFile:
        return cls.from_bytes(blob_data, extension)

    @classmethod
    def from_stdin(cls, extension: FileExtension) -> RawFile:
        data = sys.stdin.buffer.read()
        return cls.from_bytes(data, extension)

    @classmethod
    def from_ftp(
        cls,
        host: str,
        filepath: str,
        username: str = "",
        password: str = "",
        extension: Optional[FileExtension] = None,
    ) -> RawFile:
        import ftplib

        ftp = ftplib.FTP(host)
        ftp.login(user=username, passwd=password)
        data = bytearray()
        ftp.retrbinary(f"RETR {filepath}", data.extend)
        ftp.quit()
        if not extension:
            extension = FileExtension(Path(filepath).suffix.lstrip("."))
        return cls(contents=bytes(data), extension=extension)

    def save_to_file(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            f.write(self.contents)

    def get_size(self) -> int:
        return len(self.contents)

    def compute_md5(self) -> str:
        md5 = hashlib.md5()
        md5.update(self.contents)
        return md5.hexdigest()

    def compute_sha256(self) -> str:
        sha256 = hashlib.sha256()
        sha256.update(self.contents)
        return sha256.hexdigest()

    def get_mime_type(self) -> str:
        mime_type, _ = mimetypes.guess_type(f"file.{self.extension}")
        return mime_type or "application/octet-stream"

    def compress(self) -> RawFile:
        import gzip

        compressed_data = gzip.compress(self.contents)
        return RawFile(contents=compressed_data, extension=self.extension)

    def decompress(self) -> RawFile:
        import gzip

        decompressed_data = gzip.decompress(self.contents)
        return RawFile(contents=decompressed_data, extension=self.extension)

    async def read_async(self) -> bytes:
        return self.contents

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass  # Nothing to close since we're using bytes

    def __del__(self):
        pass  # No cleanup needed

    @staticmethod
    def _get_extension_from_content_type(content_type: str) -> Optional[FileExtension]:
        content_type_map = {
            "application/pdf": FileExtension.PDF,
            "application/json": FileExtension.JSON,
            "image/png": FileExtension.PNG,
            "image/jpeg": FileExtension.JPEG,
            "image/jpg": FileExtension.JPG,
            "text/html": FileExtension.HTML,
            "text/plain": FileExtension.TXT,
            # Add more mappings as needed
        }
        return content_type_map.get(content_type, None)
