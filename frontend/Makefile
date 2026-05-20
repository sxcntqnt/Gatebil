.PHONY: build run test lint docker-up docker-down fmt vet

APP=kyc-service
CMD=./cmd/server

build:
	go build -o bin/$(APP) $(CMD)

run:
	go run $(CMD)

test:
	go test ./... -v -race -count=1

lint:
	golangci-lint run ./...

fmt:
	gofmt -w .
	goimports -w .

vet:
	go vet ./...

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down -v

# Quick smoke-test against a running local instance
smoke:
	@echo "--- Submit KYC job ---"
	curl -s -X POST http://localhost:8080/api/v1/kyc/submit \
	  -H "Content-Type: application/json" \
	  -d '{"user_id":"u-001","country_code":"KE","id_type":"NATIONAL_ID","id_number":"12345678","first_name":"Jane","last_name":"Doe","tier":"kyc_light"}' | jq .

	@echo "--- Health ---"
	curl -s http://localhost:8080/healthz | jq .
