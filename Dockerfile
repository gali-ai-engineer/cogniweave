FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY cogniweave ./cogniweave
RUN pip install --no-cache-dir ".[anthropic,openai]"

# Long-term memory persists across container runs when /app/.cogniweave is mounted.
VOLUME ["/app/.cogniweave"]
ENTRYPOINT ["cogniweave"]
CMD ["run", "--pattern", "hierarchical", "--usage"]
