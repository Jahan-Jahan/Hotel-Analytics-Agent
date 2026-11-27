.PHONY: install db-role test test-unit ask chat eval ui docker-build docker-up docker-down

install:
	pip install -r requirements-dev.txt

# Create/refresh the read-only DB role. Re-run after `make sql` in Hotel-Booking-Demand
# (that script recreates the bookings view and drops its grants).
db-role:
	python -m hotel_agent.setup_db

test:
	pytest -q

test-unit:
	pytest -q -m "not integration"

chat:
	python -m hotel_agent.cli -v

ask:
	python -m hotel_agent.cli -v --ask "$(Q)"

eval:
	python -m hotel_agent.evaluation

ui:
	streamlit run app.py

docker-build:
	docker build -t hotel-analytics-agent .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
