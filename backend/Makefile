.PHONY: help run test vet build swagger swagger-down swagger-config

help:
	@echo "run            Start the backend using local .env and environment"
	@echo "test           Run Go tests"
	@echo "vet            Run Go static checks"
	@echo "build          Build the backend into bin/"
	@echo "swagger        Start Swagger UI on http://localhost:8081"
	@echo "swagger-down   Stop Swagger UI"
	@echo "swagger-config Validate the Compose configuration"

run:
	go run .

test:
	go test ./...

vet:
	go vet ./...

build:
	go build -o bin/ .

swagger:
	docker compose up -d swagger

swagger-down:
	docker compose down

swagger-config:
	docker compose config --quiet
