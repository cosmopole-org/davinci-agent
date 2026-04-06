# Point metadata contract for dynamic tool discovery

Davinci discovers tool capabilities from `metadata.tools` returned by Caspar point
listing APIs (`/points/listApps`), across:

- `machines`
- `apps`
- `programs`

For MCP-style machines, metadata should follow this shape:

```json
{
  "isMcp": true,
  "tools": [
    {
      "name": "set",
      "desc": "save a key value data into redis",
      "args": {
        "key": {
          "type": "STRING",
          "desc": "the key of value to be inserted into redis"
        },
        "value": {
          "type": "STRING",
          "desc": "the value to be inserted into redis"
        }
      }
    }
  ]
}
```

For Davinci routing, keep the MCP structure above and add routing fields
(`tool_id`, `vm_name`, `request_point`, `response_point`, plus optional
`categories` + `routes` map).

## Required tool fields

- `tool_id`
- `vm_name`
- `description`
- `request_point`
- `response_point`

## Recommended fields

- `prompt`
- `machine_id`
- `image_name`
- `container_name`
- `risk_level` (`low` | `medium` | `high`)
- `requires_network` (`true` | `false`)
- `tools` (list)

## Function metadata format (`tools` entries)

```json
{
  "name": "search",
  "desc": "search the web with a text query",
  "args": {
    "query": {
      "type": "STRING",
      "desc": "web search query"
    }
  }
}
```

## Machine metadata example (shape returned by point listing)

```json
{
  "metadata": {
    "tools": [
      {
        "tool_id": "web_search",
        "vm_name": "web-vm",
        "description": "Search web",
        "request_point": "tool::web_search::request",
        "response_point": "tool::web_search::response",
        "prompt": "Use to gather fresh web results and citations.",
        "categories": ["research"],
        "routes": {
          "research": "search"
        },
        "tools": [
          {
            "name": "search",
            "desc": "search the public web for relevant sources",
            "args": {
              "query": {
                "type": "STRING",
                "desc": "web search query"
              }
            }
          }
        ]
      }
    ]
  }
}
```
