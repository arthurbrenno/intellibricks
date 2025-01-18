# Changelog

## v0.4.0

- feat: new (experimental) IntellibricksFileParser.
- refactor: changed ParsedDocument class to ParsedFile, which represents a more generalized way to represent "non-document" files, like images.
- refactor: changed "pages", inside ParsedFile class to "sections" which generalises sections of any kind of file in a more meaningful way.
- perf: made in the previous version, but important to notice that all msgspec-jsonschema conversions are now faster.
