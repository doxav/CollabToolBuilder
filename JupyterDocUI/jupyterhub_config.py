# jupyterhub_config.py

from jupyter_client.localinterfaces import public_ips
import yaml


# # Load collaboration configuration from file
# with open('/srv/jupyterhub/collaboration_config.yml', 'r') as f:
#     project_config = yaml.safe_load(f)


# JupyterHub Configuration
c.JupyterHub.ip = '0.0.0.0'
c.JupyterHub.port = 8000
c.JupyterHub.hub_ip = public_ips()[0]
c.JupyterHub.proxy_api_ip = public_ips()[0]
c.JupyterHub.log_level = 'DEBUG'


# Use DockerSpawner to spawn user servers in Docker containers
c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'

# Configure the DockerSpawner
# jupyterhub_config.py

c.DockerSpawner.image = 'jupyter/base-notebook:latest'
c.DockerSpawner.network_name = 'jupyterhub-network'
c.DockerSpawner.use_internal_ip = True

c.DockerSpawner.extra_host_config = {"network_mode": "jupyterhub-network"}

# Mount the user's notebook directory into the Docker container
c.DockerSpawner.notebook_dir = '/home/jovyan/work'

# Authenticator configuration using Native Authenticator
# c.JupyterHub.authenticator_class = 'nativeauthenticator.NativeAuthenticator'
c.JupyterHub.authenticator_class = 'native'

# Native Authenticator configuration
c.NativeAuthenticator.create_system_users = True

# Define admin users (replace with your usernames)
c.Authenticator.admin_users = {'anwar', 'aleeza'}

# Grant admin access to users for managing collaboration
# c.JupyterHub.admin_access = True

# Pre/pare to define roles and groups
c.JupyterHub.load_roles = []
c.JupyterHub.load_groups = {'box-project': 'anwar,aleeza'}

# Iterate through collaborations and create users, assign permissions
# for project_name, project in project_config['projects'].items():
#     members = project.get('members', [])
#     c.JupyterHub.load_groups[project_name] = members
    
#     collab_user = f'{project_name}-collab'
#     c.JupyterHub.load_groups['collaborative'].append(collab_user)
    
#     c.JupyterHub.load_roles.append(
#         {
#             'name': f'collab-access-{project_name}',
#             'scopes': [
#                 f'access:servers!user={collab_user}',
#                 f'admin:servers!user={collab_user}',
#                 'admin-ui',
#                 f'list:users!user={collab_user}',
#             ],
#             'groups': [project_name],
#         }
#     )

c.JupyterHub.load_groups = {
    'admins': ['admin'],
    'normal-users': ['user1', 'user2']
}

c.JupyterHub.load_roles = [
    {
        'name': 'admin-role',
        'scopes': ['admin:servers', 'admin-ui'],
        'groups': ['admins']
    },
    {
        'name': 'normal-user-role',
        'scopes': ['access:servers', 'list:users'],
        'groups': ['normal-users']
    }
]

# def pre_spawn_hook(spawner):
#     group_names = {group.name for group in spawner.user.groups}
    
#     # Grant roles and scopes
#     for role in c.JupyterHub.load_roles:
#         if set(role['groups']).intersection(set(group_names)):
#             spawner.log.info(f'Granting role {role["name"]} to user {spawner.user.name}')
#             user.roles.append(role['name'])
#             spawner.server.scopes.extend(role['scopes'])
    
#     # Enable Collaboration
#     if 'collaborative' in group_names:
#         spawner.log.info(f'Enabling RTC for user {spawner.user.name}')
#         spawner.args.append('--LabApp.collaborative=True')

# Enable Real-Time Collaboration on Collaborative Servers
def pre_spawn_hook(spawner):
    group_names = {group.name for group in spawner.user.groups}
    if 'collaborative' in group_names:
        spawner.log.info(f'Enabling RTC for user {spawner.user.name}')
        spawner.args.append('--LabApp.collaborative=True')


c.JupyterHub.allow_named_servers = True
c.JupyterHub.named_server_limit_per_user = 5

c.Spawner.pre_spawn_hook = pre_spawn_hook
c.Application.log_level = 'DEBUG'
c.Spawner.http_timeout = 60  # Set to a higher value, e.g., 60 seconds
c.Spawner.args = ['--allow-root']