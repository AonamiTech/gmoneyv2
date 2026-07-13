import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "GMoney · Bill Evidence Studio",
  description: "Evidence-grounded hospital bill extraction",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
