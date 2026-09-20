import i18next from "i18next";
import { initReactI18next } from "react-i18next";

import { useUi } from "../state/ui";
import en from "./en.json";
import ru from "./ru.json";

/**
 * Every user-visible string of the window comes from here (A §7). Russian is the wording the
 * mock-ups were approved in and the fallback for anything English has not caught up with;
 * escaping is off because React escapes what it renders and i18next would otherwise turn a
 * «×» or a quote into an entity.
 *
 * The language lives in the UI store, which owns what is remembered between sessions; this
 * module only follows it, so nothing has to import i18next to change the language.
 */
void i18next.use(initReactI18next).init({
  resources: { ru: { translation: ru }, en: { translation: en } },
  lng: useUi.getState().language,
  fallbackLng: "ru",
  interpolation: { escapeValue: false },
});

useUi.subscribe((state) => {
  if (state.language !== i18next.language) {
    void i18next.changeLanguage(state.language);
  }
});

export default i18next;
