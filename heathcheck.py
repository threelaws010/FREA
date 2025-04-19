import docker

def check_container_health():
    client = docker.from_env()
    
    print("\nDocker Container Health Status:\n" + "-"*40)

    for container in client.containers.list():
        name = container.name
        status = container.status

        try:
            health = container.attrs['State'].get('Health', {}).get('Status', 'no healthcheck')
        except KeyError:
            health = 'unknown'

        print(f"{name}: {status} | Health: {health}")

if __name__ == "__main__":
    check_container_health()
