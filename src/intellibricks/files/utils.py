import magic


def detect_mime_type(content: bytes) -> str:
    try:
        return magic.Magic(mime=True).from_buffer(content)
    except Exception:
        return _detect_mime_type_manually(content)


def _detect_mime_type_manually(content: bytes) -> str:
    # Ordered by category and signature specificity (longer/more specific first)
    signature_map = [
        # Images
        ("image/jp2", [(0, bytes.fromhex("00 00 00 0C 6A 50 20 20"))]),
        ("image/png", [(0, bytes.fromhex("89 50 4E 47 0D 0A 1A 0A"))]),
        ("image/tiff", [(0, b"II*\x00"), (0, b"MM\x00*")]),
        ("image/webp", [(8, b"WEBP")]),  # After 'RIFF' header
        ("image/jpeg", [(0, bytes.fromhex("FF D8 FF"))]),
        ("image/gif", [(0, b"GIF87a"), (0, b"GIF89a")]),
        ("image/bmp", [(0, b"BM")]),
        ("image/x-icon", [(0, bytes.fromhex("00 00 01 00"))]),
        # Documents
        ("application/pdf", [(0, b"%PDF")]),
        ("application/msword", [(0, bytes.fromhex("D0 CF 11 E0 A1 B1 1A E1"))]),
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            [(0, b"PK\x03\x04"), (30, b"word/")],
        ),
        ("application/vnd.ms-excel", [(0, bytes.fromhex("D0 CF 11 E0 A1 B1 1A E1"))]),
        (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            [(0, b"PK\x03\x04"), (30, b"xl/")],
        ),
        (
            "application/vnd.ms-powerpoint",
            [(0, bytes.fromhex("D0 CF 11 E0 A1 B1 1A E1"))],
        ),
        (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            [(0, b"PK\x03\x04"), (30, b"ppt/")],
        ),
        ("application/rtf", [(0, b"{\\rtf")]),
        # Archives
        ("application/zip", [(0, b"PK\x03\x04")]),
        ("application/x-rar-compressed", [(0, b"Rar!\x1a\x07\x00")]),
        ("application/x-7z-compressed", [(0, bytes.fromhex("37 7A BC AF 27 1C"))]),
        ("application/gzip", [(0, bytes.fromhex("1F 8B"))]),
        ("application/x-xz", [(0, bytes.fromhex("FD 37 7A 58 5A 00"))]),
        ("application/x-bzip2", [(0, b"BZh")]),
        # Audio/Video
        ("audio/mpeg", [(0, b"ID3")]),
        ("audio/flac", [(0, b"fLaC")]),
        ("audio/ogg", [(0, b"OggS")]),
        ("audio/x-wav", [(8, b"WAVE")]),  # After 'RIFF' header
        ("video/mp4", [(4, b"ftyp")]),
        ("video/x-msvideo", [(8, b"AVI ")]),  # After 'RIFF' header
        ("video/quicktime", [(4, b"ftypqt")]),
        # System/Executables
        ("application/x-msdownload", [(0, b"MZ")]),
        ("application/vnd.ms-cab-compressed", [(0, b"MSCF")]),
        ("application/x-shockwave-flash", [(0, b"FWS"), (0, b"CWS")]),
        # Databases
        ("application/vnd.sqlite3", [(0, b"SQLite format 3\x00")]),
        # Text Formats
        ("text/xml", [(0, b"<?xml")]),
        ("application/json", [(0, b"{"), (0, b"[")]),
    ]

    # Check magic numbers
    for mime_type, signatures in signature_map:
        for offset, sig in signatures:
            if len(content) >= offset + len(sig):
                if content[offset : offset + len(sig)] == sig:
                    return mime_type

    # Special text detection
    text_mimes = [
        (b"\xef\xbb\xbf", "text/plain"),  # UTF-8 BOM
        (b"\xfe\xff", "text/plain"),  # UTF-16 BE
        (b"\xff\xfe", "text/plain"),  # UTF-16 LE
        (b"\x00\x00\xfe\xff", "text/plain"),  # UTF-32 BE
        (b"\xff\xfe\x00\x00", "text/plain"),  # UTF-32 LE
    ]

    for bom, mime in text_mimes:
        if content.startswith(bom):
            return mime

    try:
        content.decode("utf-8")
        return "text/plain"
    except UnicodeDecodeError:
        pass

    return "application/octet-stream"
