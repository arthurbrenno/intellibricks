from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from google.genai import types


from google.genai import types


def schema_to_genai_schema(schema_dict: dict[str, Any]) -> types.Schema:
    from google.genai import types

    def map_type(json_type: Optional[str]) -> types.Type:
        if json_type == "object":
            return "OBJECT"
        elif json_type == "array":
            return "ARRAY"
        elif json_type == "string":
            return "STRING"
        elif json_type == "integer":
            return "INTEGER"
        elif json_type == "number":
            return "NUMBER"
        elif json_type == "boolean":
            return "BOOLEAN"
        return "TYPE_UNSPECIFIED"

    def convert(schema: dict[str, Any]) -> types.Schema:
        schema_type = schema.get("type")
        converted_type = map_type(schema_type) if schema_type else None

        title = schema.get("title")
        description = schema.get("description")
        default = schema.get("default")
        enum = schema.get("enum")
        pattern = schema.get("pattern")
        format_ = schema.get("format")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")

        # Convert integers to strings
        min_length_str = str(schema["minLength"]) if "minLength" in schema else None
        max_length_str = str(schema["maxLength"]) if "maxLength" in schema else None
        min_items_str = str(schema["minItems"]) if "minItems" in schema else None
        max_items_str = str(schema["maxItems"]) if "maxItems" in schema else None
        min_props_str = (
            str(schema["minProperties"]) if "minProperties" in schema else None
        )
        max_props_str = (
            str(schema["maxProperties"]) if "maxProperties" in schema else None
        )

        properties_schema = None
        if "properties" in schema and isinstance(schema["properties"], dict):
            properties_schema = {
                prop_key: convert(prop_schema)
                for prop_key, prop_schema in schema["properties"].items()
            }

        items_schema = None
        if "items" in schema:
            if isinstance(schema["items"], dict):
                items_schema = convert(schema["items"])
            elif isinstance(schema["items"], list) and len(schema["items"]) == 1:
                items_schema = convert(schema["items"][0])

        required = schema.get("required")
        example = schema.get("example")

        # Construct the argument dictionary
        schema_args = {
            "type": converted_type,
            "title": title,
            "description": description,
            "default": default,
            "enum": enum,
            "format": format_,
            "pattern": pattern,
            "minimum": minimum,
            "maximum": maximum,
            "min_length": min_length_str,
            "max_length": max_length_str,
            "min_items": min_items_str,
            "max_items": max_items_str,
            "min_properties": min_props_str,
            "max_properties": max_props_str,
            "required": required,
            "example": example,
        }

        # Only set items if it's not None
        if items_schema is not None:
            schema_args["items"] = items_schema

        # Only set properties if it's not None
        if properties_schema is not None:
            schema_args["properties"] = properties_schema

        return types.Schema(**schema_args)

    return convert(schema_dict)
