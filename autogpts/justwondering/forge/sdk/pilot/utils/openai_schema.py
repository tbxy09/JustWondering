import jsonschema
# Define the schema
openai_schema = {
    "type" : "object",
    "properties" : {
        "choices" : {
            "type" : "array",
            "items": {
                "type" : "object",
                "properties" : {
                    "message" : {
                        "type" : "object",
                        "properties" : {
                            "content" : {"type" : "string"}
                        },
                        "required": ["content"]
                    }
                },
                "required": ["message"]
            }
        }
    },
    "required": ["choices"]
}