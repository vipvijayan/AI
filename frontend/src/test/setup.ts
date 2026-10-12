import '@testing-library/jest-dom'
import { vi, expect } from 'vitest'

vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => ({
      post: vi.fn(),
      get: vi.fn(),
    })),
  },
}))