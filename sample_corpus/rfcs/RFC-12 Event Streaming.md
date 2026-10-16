---
title: "RFC-12: Event Streaming for Atlas"
tags: [rfc, platform]
---
# RFC-12: Event Streaming for Atlas

Author: Priya Raman. Status: accepted.

## Proposal

Adopt Apache Kafka as the backbone of [[Project Atlas]] instead of RabbitMQ.

## Rationale

Kafka was chosen over RabbitMQ because it supports replay of historical events and sustains much higher throughput. RabbitMQ remains in use for small task queues.
