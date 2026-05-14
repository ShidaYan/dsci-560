# The Coach

The Coach is a mobile-style React prototype for a social fitness app that combines group accountability with AI-assisted coaching. The current build focuses on a single-page, app-like experience with community chat, workout sessions, a weekly training plan, and a member profile view.

Built with React and Vite, the app is designed as a product prototype rather than a production-ready service. It demonstrates how an AI coach can support beginner-friendly fitness communities through chat guidance, visual form feedback, and natural-language session planning.

## What the App Includes

### 1. Community Feed
- Displays recent activity from group members such as runs, walks, swims, and strength sessions.
- Shows lightweight social interactions including likes and comment counts.
- Uses seeded sample content to simulate an active fitness community.

### 2. Group Chat with AI Coach
- Includes a shared chat room for members and the AI coach.
- Supports text conversations with the coach for fitness, recovery, and nutrition guidance.
- Accepts uploaded images and sends them to the AI for visual feedback such as form checks.
- Includes simulated member messages to make the prototype feel more like a live group environment.

### 3. Sessions Hub
- Lists structured group workout sessions such as Hyrox, running, CrossFit, and tennis.
- Allows filtering by session type.
- Supports joining and leaving sessions.
- Includes session detail pages with attendee lists, workout plans, and session-specific chat threads.
- Persists session data in localStorage so demo changes remain after refresh.

### 4. AI Session Organizer
- Lets users describe a workout event in natural language.
- Calls the OpenAI Chat Completions API in JSON mode to convert user input into a structured session draft.
- Generates a draft containing session type, date, time, location, capacity, and workout plan.
- Creates a new session directly from the AI-generated draft once the required fields are present.

### 5. Weekly Training Plan
- Shows a simple beginner-oriented weekly plan.
- Includes progress tracking and expandable day-by-day activities.
- Highlights the current day to anchor the user's routine.

### 6. Profile View
- Displays basic member stats, achievements, and settings-style actions.
- Reinforces the product direction as a community fitness app, not just a chatbot.

## Tech Stack

- React 18
- Vite 5
- Plain inline styling in JSX
- Browser localStorage for session persistence
- OpenAI Chat Completions API for AI chat and session planning

## Project Structure

The project is intentionally small and centered around a single main component file.

```text
.
|-- index.html
|-- package.json
|-- vite.config.js
|-- src/
|   |-- App.jsx
|   `-- main.jsx
`-- README.md
```

### Key Files
- `src/App.jsx`: contains the full prototype UI, seeded data, chat flows, session flows, and AI integrations.
- `src/main.jsx`: mounts the React app.
- `index.html`: app shell and page title.
- `package.json`: scripts and dependencies.

## Getting Started

### Prerequisites
- Node.js 18 or newer is recommended.
- npm is used in the examples below.

### Install Dependencies

```bash
npm install
```

### Start the Development Server

```bash
npm run dev
```

Vite will print a local development URL, typically:

```text
http://localhost:5173
```

### Create a Production Build

```bash
npm run build
```

## Available Scripts

- `npm run dev`: starts the Vite development server.
- `npm run build`: creates a production build in the dist output.

## How the Prototype Works

### Navigation Model
The UI behaves like a mobile app contained in a phone-sized frame. The bottom tab bar switches between:

- Feed
- Chat
- Sessions
- Plan
- Profile

The default landing view is the Sessions tab.

### Seeded Demo Data
The prototype ships with predefined:

- member profiles
- activity feed items
- chat history
- workout sessions
- training plan entries

This makes the app usable immediately without authentication or backend setup.

### Session Persistence
Session data is stored in browser localStorage. That means actions such as creating a session, joining one, or chatting inside a session remain visible after page refresh in the same browser.

## AI Integration

The app currently makes direct client-side requests to the OpenAI API for two flows:

### Coach Chat
- General fitness Q and A
- Beginner-friendly coaching replies
- Optional image-based feedback when a user uploads a photo

### Session Organizer
- Natural-language event planning
- Structured JSON responses
- Auto-filled session drafts before creation

## Important Security Note

The current prototype stores the OpenAI API configuration directly in `src/App.jsx`. That is acceptable only for a quick local prototype and should not be used in production.

For any real deployment, move API access behind a backend or serverless function and load secrets from environment variables instead of shipping them in the client bundle.

## Current Limitations

- No authentication or user accounts
- No backend or database
- No real-time messaging service
- No production-grade API secret handling
- Most UI state is local to the browser
- Styling is implemented inline rather than through a reusable design system
- The entire prototype currently lives in a single large React component file

## Suggested Next Improvements

If this prototype is going to evolve into a real product, the next practical steps would be:

1. Move OpenAI calls to a secure backend.
2. Split `src/App.jsx` into reusable components and feature modules.
3. Add environment-based configuration.
4. Introduce a proper data model and API layer.
5. Add tests for core UI behaviors and session flows.
6. Replace seeded data with persisted user and session data.

## Status

This repository is best understood as a polished frontend prototype for exploring the product direction of an AI-powered fitness community app.
