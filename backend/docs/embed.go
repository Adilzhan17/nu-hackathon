package docs

import _ "embed"

// OpenAPI embeds the API contract served by GET /openapi.yaml.
//
//go:embed openapi.yaml
var OpenAPI []byte
