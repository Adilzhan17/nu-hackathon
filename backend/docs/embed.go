package docs

import _ "embed"

// OpenAPI is the same contract mounted by the local Swagger UI.
//
//go:embed openapi.yaml
var OpenAPI []byte
