# DeepFellow Infra WebUI

React-based web interface for managing DeepFellow services and models.

## Features

- **Admin Authentication**: Simple admin API key-based authentication
- **Services Management**: View, install, and uninstall services
- **Models Management**: View, install, and uninstall models for each service
- **Dynamic Forms**: Auto-generated forms based on service/model specs from the API
- **Filtering**: Search and filter models by name, type, and installation status

## Tech Stack

- **React 19** - UI framework
- **Vite** - Build tool
- **TanStack Router** - File-based routing
- **TanStack Query** - Server state management
- **TanStack Form** - Form handling
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components

## Project Structure

```
src/
├── components/
│   ├── ui/              # shadcn/ui components
│   ├── DynamicFormModal.tsx  # Dynamic form generator
│   ├── ServicesList.tsx      # Services list view
│   └── ServiceModels.tsx     # Models list view with filtering
├── deepfellow/
│   ├── client.ts        # API client
│   └── types.ts         # TypeScript types
├── hooks/
│   └── use-auth.ts      # Authentication hook
├── lib/
│   └── auth.ts          # Auth utilities
└── routes/
    ├── __root.tsx       # Root layout
    ├── sign-in.tsx      # Login page
    └── dashboard/
        ├── index.tsx    # Services list page
        └── services/
            └── $serviceId.tsx  # Service models page
```

## Development

```bash
# Install dependencies
npm install

# Start dev server (runs on port 3000)
npm run dev

# Build for production
npm run build

# Build and copy to static folder
npm run buildx
```

## API Endpoints Used

- `GET /admin/services` - List all services
- `GET /admin/services/{id}` - Get service details and spec
- `POST /admin/services/{id}` - Install a service
- `DELETE /admin/services/{id}` - Uninstall a service
- `GET /admin/services/{id}/models` - List models for a service
- `GET /admin/services/{id}/models/_?model_id={modelId}` - Get model details and spec
- `POST /admin/services/{id}/models/_?model_id={modelId}` - Install a model
- `DELETE /admin/services/{id}/models/_?model_id={modelId}` - Uninstall a model

## Environment Variables

- `VITE_DF_SERVER_URL` - Backend server URL (default: `http://localhost:8000`)

## Views

### 1. Login (`/sign-in`)
Simple login form requesting admin API key, which is stored in localStorage.

### 2. Services List (`/dashboard`)
Displays all available services with:
- Installation status badge
- Size information
- Installed configuration values
- Install/Uninstall/Models buttons

### 3. Service Models (`/dashboard/services/:serviceId`)
Displays all models for a service with:
- Search by name filter
- Type filter (TTS, STT, LLM, Embedding, LORA)
- Installation status filter
- Model cards showing type, size, and configuration
- Install/Uninstall buttons

## Notes

- All forms are dynamically generated based on the spec returned from the API
- Password fields are masked in the display
- The admin API key is required for all API requests
- Invalid API key will result in 401/403 errors
