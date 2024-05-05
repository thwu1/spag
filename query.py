from openai import AsyncOpenAI
import asyncio
import time

num_clients = 8
model_name = "meta-llama/Llama-2-7b-hf"
BATCH_SIZE = 1250
BATCH_TIMEOUT = 1

test_num = 10000


class Client:
    def __init__(self, base_url, api_key):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._is_free = True

    async def get_batch_completion(self, prompts):
        # tasks = [self.get_completion(prompt) for prompt in prompts]
        self._is_free = False
        completion = await self.client.completions.create(
            model=model_name, prompt=prompts
        )
        self._is_free = True
        return completion.choices

    async def is_free(self):
        return self._is_free


class ClientGroup:
    def __init__(self, num_clients, model_name):
        self.clients = [
            Client(
                base_url=f"http://localhost:{8000 + i}/v1",
                api_key="token-abc123",
            )
            for i in range(num_clients)
        ]
        self.num_clients = num_clients
        self.current_client = 0
        self.model_name = model_name
        self.request_queue = asyncio.Queue()
        self.results = []
        self.active_tasks = 0

    async def add_request(self, prompt):
        await self.request_queue.put(prompt)

    async def process_requests(self, total_requests):
        current_batch = []
        batch_start_time = None

        while True:
            try:
                # Wait for the next request or timeout
                prompt = await asyncio.wait_for(
                    self.request_queue.get(), timeout=BATCH_TIMEOUT
                )
                current_batch.append(prompt)
                self.request_queue.task_done()
                # Start the timer when the first request is received
                if batch_start_time is None:
                    batch_start_time = time.time()

            except asyncio.TimeoutError:
                # Timeout reached with a partially filled batch
                if current_batch:
                    await self.process_batch(current_batch)
                    current_batch = []
                    batch_start_time = None

            # Check if the batch is full or the timeout has been reached
            if len(current_batch) >= BATCH_SIZE * self.num_clients or (
                batch_start_time is not None
                and time.time() - batch_start_time >= BATCH_TIMEOUT
            ):
                await self.process_batch(current_batch)
                # self.request_queue.task_done()
                current_batch = []
                batch_start_time = None

            # Check if the queue is empty
            if len(self.results) == total_requests:
                break

    async def process_batch(self, batch):
        # Process the batch of requests concurrently
        free_clients = [client for client in self.clients if await client.is_free()]
        num_requests = len(batch)
        requests_per_client = (num_requests // len(free_clients)) + 1

        self.active_tasks += num_requests
        tasks = [
            client.get_batch_completion(
                batch[requests_per_client * i : requests_per_client * (i + 1)]
            )
            for i, client in enumerate(free_clients)
        ]
        results = await asyncio.gather(*tasks)
        results = [item for sublist in results for item in sublist]
        self.results.extend(results)
        self.active_tasks -= num_requests

    async def run(self, prompts):
        start = time.time()
        for prompt in prompts:
            await self.add_request(prompt)
        end = time.time()
        print(f"Added {len(prompts)} requests in {end - start} seconds")

        total_requests = len(prompts)

        # Process requests and collect results
        await self.process_requests(total_requests)
        results = [result.text for result in self.results]
        self.results = []

        return results


async def main(questions):
    # questions = ["Hello, explain the concept of a neural network."] * test_num
    group = ClientGroup(num_clients, model_name)
    start = time.time()
    completions = await group.run(questions)
    end = time.time()
    print(f"{(end - start)/len(questions)} seconds per completion")
    print("total time: ", end - start)
    print(completions[0])
    print(len(completions), len(questions))
    assert len(completions) == len(questions)


if __name__ == "__main__":
    questions = []
    with open("/scratch/tianhao/spag/data/all_target_words.txt", "r") as f:
        for item in f:
            questions.append(item.strip())
    
    asyncio.run(main(questions))