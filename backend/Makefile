.PHONY: help swagger swagger-down swagger-config

help:
	@echo "swagger        Start standalone Swagger UI on http://localhost:8081"
	@echo "swagger-down   Stop standalone Swagger UI"
	@echo "swagger-config Validate the Compose configuration"
	@echo "Application build/run targets will be added with implementation."

swagger:
	docker compose up -d swagger

swagger-down:
	docker compose down

swagger-config:
	docker compose config --quiet
