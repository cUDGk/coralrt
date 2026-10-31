# From pycoral 2.0 and libcoral (Copyright Google LLC, Apache License 2.0), rewritten; see NOTICE.
"""pycoral.pipeline.pipelined_model_runner on coralrt: runs a model segmented for several Edge TPUs, one thread per
segment, as libcoral's PipelinedModelRunner does: tensors pass from segment to segment by name (a tensor goes on
until the last segment that reads it), the queues between segments hold one request, and an empty push stops
the pipeline once everything pushed before it is done.
"""
import collections
import threading

import numpy as np


def _get_names(details):
  return {d['name'] for d in details}


class _Queue:
  """libcoral's WaitQueue: push blocks while full; wait returns None once stopped and empty."""

  def __init__(self, max_size=0):
    self.items, self.max_size, self.stopped = collections.deque(), max_size, False
    self.cond = threading.Condition()

  def push(self, item):
    with self.cond:
      while self.max_size and len(self.items) >= self.max_size:
        self.cond.wait()
      self.items.append(item)
      self.cond.notify_all()

  def wait(self):
    with self.cond:
      while not self.items:
        if self.stopped:
          return None
        self.cond.wait()
      item = self.items.popleft()
      self.cond.notify_all()
      return item

  def set_max_size(self, size):
    with self.cond:
      self.max_size = size
      self.cond.notify_all()

  def stop(self):
    with self.cond:
      self.stopped = True
      self.cond.notify_all()


class PipelinedModelRunner:
  """Manages the model pipeline: the interpreters of the segments, in order, each with allocate_tensors() done."""

  def __init__(self, interpreters):
    self._runner = None
    if not interpreters:
      raise ValueError('At least one interpreter expected')
    # the inputs of interpreter i must be outputs of the ones before it (or the pipeline's inputs)
    prev_outputs = _get_names(interpreters[0].get_input_details())
    for index, interpreter in enumerate(interpreters):
      inputs = _get_names(interpreter.get_input_details())
      if not inputs.issubset(prev_outputs):
        raise ValueError('Interpreter {} can not get its input tensors'.format(index))
      prev_outputs.update(_get_names(interpreter.get_output_details()))
    self._interpreters = interpreters
    self._input_types = {d['name']: d['dtype'] for d in interpreters[0].get_input_details()}
    self._output_shapes = {d['name']: d['shape'] for d in interpreters[-1].get_output_details()}
    n = len(interpreters)
    self._consumers = collections.Counter(name for i in interpreters for name in _get_names(i.get_input_details()))
    self._queues = [_Queue(1 if 0 < k < n else 0) for k in range(n + 1)]
    self._errors = [None] * n
    self._lock = threading.Lock()
    self._on = True
    self._threads = [threading.Thread(target=self._run, args=(k,), daemon=True) for k in range(n)]
    for t in self._threads:
      t.start()
    self._runner = True

  def _run(self, k):
    """SegmentRunner::RunInference: one request at a time; an error (or an empty request) stops this segment and
    passes an empty request on."""
    interpreter = self._interpreters[k]
    inputs = interpreter.get_input_details()
    names = _get_names(inputs)
    while True:
      tensors = self._queues[k].wait()
      if tensors is None:
        return
      try:
        if not tensors:
          raise RuntimeError('Empty input tensor map.')
        for d in inputs:
          interpreter.set_tensor(d['index'], tensors[d['name']][0].reshape(d['shape']))
        interpreter.invoke()
        outputs = {d['name']: [interpreter.get_tensor(d['index']).copy(), self._consumers[d['name']]]
                   for d in interpreter.get_output_details()}
      except Exception as e:  # passed on to push() / pop() as a RuntimeError, as libcoral's runner status
        self._errors[k] = 'Segment {} runner error: {}'.format(k, e)
        self._queues[k + 1].push({})
        return
      for name, entry in tensors.items():
        if name in names:
          entry[1] -= 1
        if entry[1] > 0:
          outputs[name] = entry
      self._queues[k + 1].push(outputs)

  def _status(self):
    for e in self._errors:
      if e:
        return e
    return None

  def _shutdown(self):
    with self._lock:
      if not self._on:
        raise RuntimeError('Pipeline was turned off before.')
      for k, t in enumerate(self._threads):
        self._queues[k].stop()
        t.join()
      self._queues[-1].stop()
      self._on = False

  def __del__(self):
    if self._runner:
      try:
        self.push({})
      except RuntimeError:
        return
      while self.pop():
        pass

  def set_input_queue_size(self, size):
    """Sets the maximum number of inputs that may wait for inference (unlimited by default)."""
    self._queues[0].set_max_size(size)

  def set_output_queue_size(self, size):
    """Sets the maximum number of results that may wait to be popped (unlimited by default)."""
    self._queues[-1].set_max_size(size)

  def push(self, input_tensors):
    """Pushes a request: {input tensor name: numpy array}. An empty dict ends the pipeline (pop() then returns None
    once the results before it are taken); pushing after that raises RuntimeError. Blocks while the input queue
    is full (see set_input_queue_size)."""
    if input_tensors and len(input_tensors) != len(self._input_types):
      raise ValueError('Expected input of length {}, but got {}'.format(len(self._input_types), len(input_tensors)))
    for key, tensor in input_tensors.items():
      input_type = self._input_types[key]
      if not isinstance(tensor, np.ndarray) or tensor.dtype != input_type:
        raise ValueError('Input should be a list of numpy array of type {}'.format(input_type))
    status = self._status()
    if status:
      try:
        self._shutdown()
      except RuntimeError:
        pass
      raise RuntimeError(status)
    if not input_tensors:
      self._shutdown()
      return
    request = {name: [np.array(t, copy=True).reshape(-1), self._consumers[name]] for name, t in input_tensors.items()}
    with self._lock:
      if not self._on:
        raise RuntimeError('Pipeline was turned off before.')
    self._queues[0].push(request)

  def pop(self):
    """Returns the next result, {output tensor name: numpy array}, blocking until it is there; None once the
    pipeline has ended and every result has been taken."""
    result = self._queues[-1].wait()
    if result is None:
      return None
    status = self._status()
    if status:
      try:
        self._shutdown()
      except RuntimeError:
        pass
      raise RuntimeError(status)
    return {d['name']: result[d['name']][0].reshape(self._output_shapes[d['name']])
            for d in self._interpreters[-1].get_output_details()}

  def interpreters(self):
    """Returns the interpreters the runner was made with."""
    return self._interpreters
