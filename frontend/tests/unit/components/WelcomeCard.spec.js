import { shallowMount } from "@vue/test-utils";
import WelcomeCard from "@/components/WelcomeCard.vue";

describe("WelcomeCard.vue", () => {
  it("Should match snapshot", () => {
    const wrapper = shallowMount(WelcomeCard);
    expect(wrapper).toMatchSnapshot();
  });
});
