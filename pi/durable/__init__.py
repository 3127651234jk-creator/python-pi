"""pi.durable: durable conversation, task, and document runtime.

Python port of the core of ``packages/durable``. Everything is committed to
storage before it is visible; a reopened session picks durable work up where
it stopped.
"""
