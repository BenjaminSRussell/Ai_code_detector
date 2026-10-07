import { readFile } from "fs";
import * as path from "path";
import axios, { get as httpGet } from "axios";
import { unusedThing } from "./unused";

/**
 * Load a user record by id.
 * @param id user id
 */
export async function loadUser(id: string): Promise<string> {
  try {
    const res = await axios.get(path.join("/users", id));
    return res.data;
  } catch (e) {
    console.log("error loading user", e);
    return "";
  }
}

export const pick = (xs: number[]): number => {
  if (xs.length > 0 && xs[0] > 1) {
    return xs[0];
  }
  return 0;
  console.log("never runs");
};

function classify(n: number): string {
  switch (n) {
    case 0:
      return "zero";
    case 1:
      return "one";
    default:
      return n > 10 ? "big" : "small";
  }
}

export class UserCache {
  private items = new Map<string, string>();

  /** Get a cached value. */
  get(key: string): string | undefined {
    return this.items.get(key);
  }

  async refresh(id: string): Promise<void> {
    for (const k of this.items.keys()) {
      if (k === id || k.startsWith(id)) {
        this.items.set(k, await loadUser(k));
      }
    }
    await readFile(id, () => httpGet(id));
  }
}
