.PHONY: help run test vet build

help:
	@echo "run            Start the backend using local .env and environment"
	@echo "test           Run Go tests"
	@echo "vet            Run Go static checks"
	@echo "build          Build the backend into bin/"

run:
	go run .

test:
	go test ./...

vet:
	go vet ./...

build:
	go build -o bin/ .
